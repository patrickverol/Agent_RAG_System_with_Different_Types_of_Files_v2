# Projeto 8 - IA Generativa Multimodal com Agentic RAG e LangGraph Para Análise Contábil
# Módulo da App

# Imports
import os
import re
import io
import base64 
import operator
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.graph import StateGraph, END

# Carrega variáveis de ambiente a partir do arquivo .env
load_dotenv()

# Recupera a chave da API do Google definida nas variáveis de ambiente
google_api_key = os.getenv("GOOGLE_API_KEY")

# Verifica se a chave da API do Google está definida; emite erro e interrompe se não estiver
if not google_api_key:
    
    # Exibe mensagem de erro no Streamlit caso a chave não esteja definida
    st.error("GOOGLE_API_KEY não definida no ambiente ou .env!")
    
    # Interrompe a execução do aplicativo
    st.stop()

# Define o caminho onde o índice FAISS será salvo
VECTORSTORE_PATH = "dsa_faiss_index_contabilidade"

# Decora a função para que o Streamlit cacheie o recurso e o carregue apenas uma vez
@st.cache_resource
def dsa_carrega_llm_vision():
    
    # Informa que o carregamento do LLM Gemini Vision começou
    print("Carregando LLM Gemini Vision...")
    try:
        
        # Cria instância do modelo ChatGoogleGenerativeAI com parâmetros especificados
        llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro-latest",
                                     google_api_key = google_api_key,
                                     temperature = 0.2)
        
        # Retorna a instância do LLM criada
        return llm
    
    except Exception as e:
        
        # Exibe mensagem de erro no Streamlit caso falhe ao carregar o modelo
        st.error(f"Erro ao carregar LLM Gemini: {e}")
        
        # Interrompe a execução do aplicativo em caso de erro
        st.stop()

# Decora a função para que o Streamlit cacheie o recurso e o carregue apenas uma vez
@st.cache_resource
def dsa_rag_retriever():

    # Informa que o carregamento do retriever do RAG de Contabilidade começou
    print("Carregando Retriever RAG Contabilidade...")
    
    # Verifica se a pasta do índice FAISS existe e não está vazia; emite aviso se falhar
    if not os.path.exists(VECTORSTORE_PATH) or not os.listdir(VECTORSTORE_PATH):
        
        # Exibe um aviso no Streamlit orientando a executar o setup caso o índice não exista
        st.warning(f"Índice RAG '{VECTORSTORE_PATH}' não encontrado. A análise usará apenas a imagem e a pergunta. Execute 'setup_rag_accounting.py' com PDFs.")
        
        # Retorna None para indicar que não há retriever disponível
        return None

    try:

        # Inicializa o modelo de embedding FastEmbed para consultas
        embedding_model = FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
        
        # Carrega o índice FAISS localmente, permitindo desserialização avançada
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization = True)
        
        # Converte o índice em um retriever configurado para buscar 3 chunks por consulta
        retriever = vector_store.as_retriever(search_kwargs = {'k': 3})
        
        # Informa que o retriever foi carregado com sucesso
        print("Retriever RAG Contabilidade carregado.")
        
        # Retorna o objeto retriever para uso nas consultas RAG
        return retriever
    
    except Exception as e:
        
        # Exibe mensagem de erro no Streamlit caso o carregamento do retriever falhe
        st.error(f"Erro ao carregar Retriever RAG Contabilidade: {e}")
        
        # Retorna None para indicar falha no carregamento
        return None

# Define o tipo de dicionário para o estado multimodal no grafo
class MultimodalGraphState(TypedDict):

    # Campo para armazenar a consulta de texto
    query: str | None

    # Campo para armazenar os bytes da imagem enviada
    image_bytes: bytes | None

    # Campo para armazenar o tipo MIME da imagem
    image_mime_type: str | None

    # Campo para armazenar o contexto recuperado via RAG
    rag_context: str | None

    # Campo para armazenar a resposta final gerada
    final_answer: str | None

# Função de nó do grafo para realizar a recuperação RAG de contabilidade
def dsa_retrieve_rag_node(state: MultimodalGraphState) -> dict:
    
    # Imprime no console o início da execução do nó RAG
    print("--- Nó: Recuperação RAG Contabilidade ---")

    # Obtém o retriever local configurado para RAG
    local_retriever = dsa_rag_retriever()

    # Extrai a consulta de texto do estado multimodal
    query = state.get("query")

    # Valor padrão para o contexto RAG caso não haja resultado
    rag_context = "Nenhum contexto RAG relevante encontrado ou RAG não disponível."

    # Se o retriever estiver disponível e houver uma consulta
    if local_retriever and query:

        try:

            # Invoca o retriever com a consulta e obtém os documentos retornados
            results = local_retriever.invoke(query)

            # Concatena o conteúdo de cada página/documento em uma string
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Se houver conteúdo recuperado, atualiza o contexto RAG
            if context:
                
                rag_context = context
                
                # Imprime a quantidade de caracteres do contexto encontrado
                print(f"  Contexto RAG Contabilidade encontrado ({len(context)} chars).")
            
            else:
                
                # Informa que não foi encontrado contexto RAG para a consulta
                print("  Nenhum contexto RAG Contabilidade encontrado para a query.")
        
        except Exception as e:

            # Em caso de erro na recuperação, imprime a exceção
            print(f"  Erro no nó RAG Contabilidade: {e}")
            
            # Atualiza o contexto com mensagem de erro
            rag_context = f"Erro ao buscar nos documentos de contabilidade: {e}"
    
    else:

         # Informa que o retriever não está disponível ou a consulta está vazia
         print("  Retriever RAG não disponível ou query vazia. Pulando busca RAG.")

    # Retorna um dicionário com o contexto RAG atualizado
    return {"rag_context": rag_context}

# Função para o nó de análise de imagens de notas fiscais
def dsa_analyze_invoice_node(state: MultimodalGraphState) -> dict:
    
    print("--- Nó: Análise Multimodal da Nota Fiscal ---")
    
    # Extrai os atributos do estado
    query = state.get("query")
    image_bytes = state.get("image_bytes")
    mime_type = state.get("image_mime_type")
    rag_context = state.get("rag_context", "Nenhum contexto adicional fornecido.")

    # Checagens para definir a resposta final no caso de não ter query ou iimagem
    if not query: return {"final_answer": "Erro: Nenhuma pergunta foi fornecida."}
    if not image_bytes or not mime_type: return {"final_answer": "Erro: Nenhuma imagem válida foi fornecida."}

    try:

        # Carrega o LLM
        llm_vision = dsa_carrega_llm_vision()

        # Gera o encode da imagem
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Prompt
        message_content = [
            {
                "type": "text",
                "text": f"""Você é um assistente de contabilidade. Sua tarefa é analisar a imagem da nota fiscal anexa e responder à pergunta do usuário. Utilize também o contexto de regras de contabilidade fornecido abaixo, se for relevante para a pergunta.

                Contexto de Regras de Contabilidade (Manuais RAG):
                ---
                {rag_context}
                ---

                Pergunta do Usuário:
                {query}

                Responda de forma clara e objetiva, baseando-se na análise da imagem e no contexto fornecido. Se a pergunta for sobre anomalias, procure por inconsistências comuns (datas, valores, cálculos, informações obrigatórias ausentes)."""
            },
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_base64}"
            }
        ]

        # Cria o objeto de mensagem
        message = HumanMessage(content = message_content)

        print(f"  Enviando para análise no Gemini Vision (imagem: {mime_type}, {len(image_bytes)} bytes)...")

        # Envia a mensagem multimodal e extrai a resposta
        response = llm_vision.invoke([message]) 
        final_answer = response.content

        print("  Análise multimodal concluída.")

        return {"final_answer": final_answer}

    except Exception as e:
        
        print(f"  Erro no nó de análise multimodal: {e}")
        import traceback
        traceback.print_exc()
        if "image" in str(e).lower():
             return {"final_answer": f"Desculpe, ocorreu um erro ao processar a imagem fornecida: {e}"}
        else:
             return {"final_answer": f"Desculpe, ocorreu um erro durante a análise multimodal: {e}"}

# Decora a função para cache de recurso no Streamlit
@st.cache_resource
def dsa_compile_multimodal_graph():

    print("Compilando o grafo multimodal...")
    
    # Cria o construtor de grafo baseado no estado multimodal definido
    graph_builder = StateGraph(MultimodalGraphState)
    
    # Adiciona nó de recuperação RAG ao grafo com a função correspondente
    graph_builder.add_node("retrieve_rag_node", dsa_retrieve_rag_node)
    
    # Adiciona nó de análise de fatura ao grafo com a função correspondente
    graph_builder.add_node("analyze_invoice_node", dsa_analyze_invoice_node)
    
    # Define o ponto de entrada inicial do grafo como o nó de recuperação RAG
    graph_builder.set_entry_point("retrieve_rag_node")
    
    # Cria aresta do nó de recuperação para o nó de análise de fatura
    graph_builder.add_edge("retrieve_rag_node", "analyze_invoice_node")
    
    # Conecta o nó de análise de fatura ao nó final do grafo
    graph_builder.add_edge("analyze_invoice_node", END)
    
    # Inicia bloco try para capturar erros na compilação do grafo
    try:
        
        # Compila o grafo multimodal em um aplicativo executável
        multimodal_app = graph_builder.compile()
        
        # Informa que a compilação do grafo foi bem-sucedida
        print("Grafo multimodal compilado com sucesso!")
        
        # Retorna o aplicativo multimodal compilado
        return multimodal_app

    except Exception as e:
        
        # Exibe mensagem de erro no Streamlit caso falhe a compilação
        st.error(f"Erro ao compilar o grafo multimodal: {e}")
        
        # Interrompe a execução do aplicativo em caso de erro crítico
        st.stop()

# Configuração da App no Streamlit
st.set_page_config(page_title="Data Science Academy", page_icon=":100:", layout="wide")
st.title("Data Science Academy - Projeto 8")
st.title("IA Generativa Multimodal com Agentic RAG e LangGraph Para Análise Contábil")
st.markdown("Faça uma pergunta sobre a nota fiscal anexada. O sistema usará os manuais de contabilidade e a análise da imagem para responder.")

# Carrega os objetos
llm_vision = dsa_carrega_llm_vision()
retriever = dsa_rag_retriever()
multimodal_app = dsa_compile_multimodal_graph()

# Cria colunas na página web e inicializa variáveis
col1, col2 = st.columns(2)
image_bytes = None
image_mime_type = None

# Elementos da coluna 1
with col1:
    st.subheader("Upload da Nota Fiscal")
    uploaded_file = st.file_uploader("Selecione a imagem da nota fiscal...", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Nota Fiscal Carregada.', use_container_width=True)
        image_bytes = uploaded_file.getvalue()
        image_mime_type = uploaded_file.type
    else:
        st.info("Aguardando upload da imagem da nota fiscal.")

# Elementos da coluna 2
with col2:
    st.subheader("Sua Pergunta")
    user_query = st.text_area("Digite sua pergunta sobre a nota fiscal:", height = 150)

    if st.button("Analisar Nota Fiscal", type="primary"):
        if user_query and image_bytes and image_mime_type:
            with st.spinner("Analisando a nota fiscal com IA..."):
                try:
                    inputs = {
                        "query": user_query,
                        "image_bytes": image_bytes,
                        "image_mime_type": image_mime_type
                    }
                    final_state = multimodal_app.invoke(inputs)

                    final_answer_raw = final_state.get("final_answer", "Não foi possível obter uma resposta.")

                    print("--- DEBUG UI: Raw Final Answer ---")
                    print(repr(final_answer_raw)) 
                    print("--- END DEBUG ---")

                    # Função para limpeza da resposta do LLM
                    def dsa_clean_llm_output(text):
                        if not isinstance(text, str):
                            return text 

                        text = text.replace('\u02ca', "'") 
                        text = text.replace('ˊ', "'")      
                        text = text.replace('\xa0', ' ')   

                        # remove caracteres invisíveis
                        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)

                        text = re.sub(r'(R\$)\s*(\d)', r'\1 \2', text)
                        text = re.sub(r'\s+(,)', r'\1', text)

                        return text.strip() 

                    final_answer_cleaned = dsa_clean_llm_output(final_answer_raw)

                    st.subheader("Resultado da Análise:")
                    st.markdown(f"<pre>{final_answer_cleaned}</pre>", unsafe_allow_html=True)

                    with st.expander("Ver Saída Crua do LLM (Debug)"):
                        st.text("Saída Crua:")
                        st.code(final_answer_raw, language=None)
                        st.text("Saída Limpa (aplicada na resposta acima):")
                        st.code(final_answer_cleaned, language=None)

                    with st.expander("Ver Contexto RAG Utilizado"):
                        st.text(final_state.get("rag_context", "Nenhum contexto RAG foi utilizado."))

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a análise: {e}")

        elif not image_bytes:
            st.warning("Por favor, faça o upload de uma imagem da nota fiscal.")
        else:
            st.warning("Por favor, digite sua pergunta sobre a nota fiscal.")

# Barra lateral com instruções de uso
st.sidebar.title("Instruções")
st.sidebar.write("""
    - Esta app utiliza LangGraph e Google Gemini Vision para análise multimodal.
    - O sistema vai "raciocinar" sobre imagens de notas fiscais, buscar contexto no RAG e responder suas dúvidas.
    - Documentos, manuais e procedimentos complementares podem ser usados para aperfeiçoar o sistema de RAG.
    - IA Generativa comete erros. SEMPRE valide as respostas.
""")

# Botão de suporte na barra lateral que exibe mensagem ao clicar
if st.sidebar.button("Suporte"):
    st.sidebar.write("Dúvidas? Envie um e-mail para: suporte@datascienceacademy.com.br")





