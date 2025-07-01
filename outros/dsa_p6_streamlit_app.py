# Projeto 6 - Multi-Agentes de IA, RAG, Roteamento, Guardrails, Observabilidade e Explicabilidade com LangGraph, LangSmith e Pydantic LogFire
# Módulo da App

# Importa o módulo os para operações de sistema como verificar arquivos e diretórios
import os

# Importa Streamlit para criar a interface web interativa
import streamlit as st

# Importa função para carregar variáveis de ambiente de um arquivo .env
from dotenv import load_dotenv

# Importa TypedDict e Literal para tipagem do estado do grafo
from typing import TypedDict, Literal

# Importa cliente Groq para chamadas ao LLM Groq
from langchain_groq import ChatGroq

# Importa FAISS para buscar documentos via RAG
from langchain_community.vectorstores import FAISS

# Importa modelo de embeddings FastEmbed para transformar textos em vetores
from langchain_community.embeddings import FastEmbedEmbeddings

# Importa ferramenta de busca DuckDuckGo
from langchain_community.tools import DuckDuckGoSearchRun

# Importa estrutura de grafo de estados e constante de fim
from langgraph.graph import StateGraph, END

# LogFire e LangSmith para Observabilidade
import logfire
from langsmith import traceable
from langsmith import Client as LangSmithClient 

########## Configuração de Variáveis de Ambiente ##########

# Ativa o paralelismo de tokenização
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

# Configuração da página da web app
st.set_page_config(page_title="Data Science Academy", page_icon=":100:", layout="centered")

# Carrega variáveis de ambiente definidas no arquivo .env
load_dotenv()

# Chave API Groq
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logfire.critical("GROQ_API_KEY não definida no ambiente ou .env!") # Usando logfire
    st.error("GROQ_API_KEY não definida no ambiente ou .env!")
    st.stop()

# Chaves API LangSmith/LangChain
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
langchain_api_key_env = os.getenv("LANGCHAIN_API_KEY")

# Token LogFire
LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")

# Configuração do Pydantic LogFire
try:
    logfire.configure() 
    print("DSA Log - Logfire configurado.") 
except Exception as e:
     print(f"DSA Log - Alerta: Falha ao configurar Logfire automaticamente: {e}")

# Agora verifica as chaves e tokens, usando st.warning para feedback na UI

if not langsmith_api_key or not langchain_api_key_env:
    st.warning("LANGSMITH_API_KEY e/ou LANGCHAIN_API_KEY não definidas. O tracing do LangSmith pode não funcionar completamente.")

if not LOGFIRE_API_KEY:
     st.warning("LOGFIRE_API_KEY não definida. Logs para Pydantic LogFire Cloud não funcionarão (a menos que outro exportador OTEL esteja configurado).")

# Verifique LANGCHAIN_TRACING_V2
if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() != "true":
    st.warning("Variável de ambiente LANGCHAIN_TRACING_V2 não está como 'true'. O tracing automático do LangGraph para o LangSmith PODE estar desativado.")

# Verifica LANGCHAIN_API_KEY especificamente para tracing
if not langchain_api_key_env:
     st.warning("Variável de ambiente LANGCHAIN_API_KEY não definida. O tracing do LangGraph para LangSmith NÃO funcionará.")

# Caminho RAG
VECTORSTORE_PATH = "dsa_faiss_index"

########## Funções Auxiliares ##########

# O carregamento do LLM consome muitos recursos e deve ser cacheado
# Decorador do Streamlit para armazenar em cache o recurso de carregamento do LLM
@st.cache_resource
# Define a função responsável por carregar o LLM que fornecerá a resposta final
def dsa_carrega_llm_resposta_final():
    
    # Imprime no console que o processo de carregamento do LLM Groq foi iniciado
    print("DSA Log - Carregando LLM Groq...")
    
    # Inicia bloco try para capturar e tratar eventuais exceções
    try:
        
        # Cria uma instância do modelo Groq usando a chave de API, o nome do modelo e a temperatura definidos
        # Este é um LLM mais robusto que será usado para a resposta final ao usuário
        llm = ChatGroq(api_key = groq_api_key, model = "llama3-70b-8192", temperature = 0.1) 
        
        # Registra no LogFire que o LLM Groq para resposta final foi carregado com sucesso
        logfire.info("LLM Groq (resposta final) carregado com sucesso.")
        
        # Retorna o objeto do LLM para uso em outras partes da aplicação
        return llm
    
    # Captura qualquer exceção lançada durante o carregamento do LLM
    except Exception as e:
        
        # Registra no LogFire o erro ocorrido, incluindo traceback para auxiliar no diagnóstico
        logfire.error("Erro ao carregar LLM final", error = str(e), exc_info = True)
        
        # Exibe uma mensagem de erro na interface Streamlit informando o usuário sobre o problema
        st.error(f"Erro ao carregar LLM: {e}")
        
        # Interrompe a execução da aplicação devido ao erro crítico no carregamento do LLM
        st.stop()

# O carregamento do retriever RAG pode consumir muitos recursos e deve ser cacheado
# Decorador do Streamlit que armazena em cache o resultado desta função
@st.cache_resource
# Define a função responsável por carregar o retriever utilizado no RAG
def dsa_carrega_retriever():
    
    # Registra no console que o processo de carregamento do retriever foi iniciado
    print("DSA Log - Carregando Retriever RAG...")
    
    # Verifica se o diretório ou arquivo do índice FAISS existe no caminho especificado
    if not os.path.exists(VECTORSTORE_PATH):
        
        # Registra um erro no LogFire informando que o índice FAISS não foi encontrado
        logfire.error("Índice FAISS não encontrado", path = VECTORSTORE_PATH)
        
        # Exibe uma mensagem de erro na interface Streamlit instruindo o usuário a executar o setup
        st.error(f"Índice FAISS não encontrado em '{VECTORSTORE_PATH}'. Execute 'dsa_p6_setup_rag.py'.")
        
        # Interrompe a execução da aplicação devido à ausência do índice FAISS
        st.stop()

    # Tenta carregar o retriever dentro de um bloco que captura exceções
    try:
        
        # Cria o modelo de embeddings FastEmbed com o nome do modelo BAAI/bge-small-en-v1.5
        embedding_model = FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
        
        # Carrega o índice FAISS local utilizando o modelo de embeddings criado
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization = True)
        
        # Converte o vector_store em um retriever, definindo o número de resultados de busca como 5
        retriever = vector_store.as_retriever(search_kwargs = {'k': 5})
        
        # Registra no LogFire que o retriever foi carregado com sucesso, indicando o caminho utilizado
        logfire.info("Retriever RAG carregado com sucesso.", path = VECTORSTORE_PATH)
        
        # Retorna o objeto retriever para uso nas operações de RAG
        return retriever
    
    # Em caso de qualquer erro durante o processo de carregamento
    except Exception as e:
        
        # Registra o erro no LogFire, incluindo detalhes da exceção e traceback
        logfire.error("Erro ao carregar Retriever RAG", path = VECTORSTORE_PATH, error = str(e), exc_info = True) 
        
        # Exibe uma mensagem de erro na interface Streamlit com a descrição da exceção
        st.error(f"Erro ao carregar Retriever RAG: {e}")
        
        # Interrompe a aplicação devido ao erro crítico no carregamento do retriever
        st.stop()

########## Funções Para os Nós do Grafo no LangGraph ##########

# Define a classe de estado do grafo (Agente de IA)
class GraphState(TypedDict):
    query: str
    source_decision: Literal["RAG", "WEB", ""]
    rag_context: str | None
    web_results: str | None
    final_answer: str | None

# Função para o nó de roteamento
@traceable(run_type = "llm", name = "Node_RouteQuery") # LangSmith Decorator
def dsa_route_query_node(state: GraphState) -> dict:
    """
    Analisa a consulta e decide a fonte de dados (RAG ou WEB).
    Atualiza 'source_decision' no estado.
    """

    # Extrai a query do estado
    query = state["query"]

    # Logfire span para agrupar logs do nó
    span = logfire.span("Executando Nó: Roteamento da Consulta", query = query)
    
    # Dentro do Span
    with span:

        # **PROMPT REFINADO COM EXEMPLOS (FEW-SHOT)**
        prompt = f"""Sua tarefa é classificar a consulta de um usuário para direcioná-la à melhor fonte de informação. As fontes são:
        1.  **RAG**: Base de conhecimento interna com documentos de suporte técnico, procedimentos específicos, configurações do nosso sistema, guias internos. Use RAG para perguntas sobre 'como fazer X em nosso sistema', 'qual a configuração de Y', 'documentação interna sobre Z'.
        2.  **WEB**: Busca geral na internet para informações sobre software de terceiros (ex: Anaconda, Python, Excel), notícias de tecnologia, erros genéricos não documentados internamente, informações muito recentes, ou qualquer coisa que não seja específica dos nossos documentos internos.

        Exemplos:
        - Consulta: "Como configuro o servidor de email interno?" -> Resposta: RAG
        - Consulta: "Qual a versão mais recente do Streamlit?" -> Resposta: WEB
        - Consulta: "Qual o procedimento para resetar a senha do sistema ABC?" -> Resposta: RAG
        - Consulta: "Como instalo o interpretador Python no Windows 11" -> Resposta: WEB
        - Consulta: "como instalar o Anaconda Python" -> Resposta: WEB

        Agora, classifique a seguinte consulta:
        Consulta do Usuário: '{query}'

        Com base na consulta, qual é a fonte mais apropriada? Responda APENAS com a palavra 'RAG' ou a palavra 'WEB'."""

        # Bloco de execução
        try:

            # **CRIA LLM DEDICADO PARA ROTEAMENTO**
            router_llm = ChatGroq(api_key = groq_api_key,
                                  model = "llama3-8b-8192", # Modelo mais rápido para roteamento
                                  temperature = 0.0)        # Baixa temperatura para decisão

            # Executa o roteador 
            response = router_llm.invoke(prompt)

            # Extrai a decisão da resposta do LLM
            raw_decision = response.content

            # Limpeza do texto de resposta
            decision = raw_decision.strip().upper().replace("'", "").replace('"', '')

            # Lógica de decisão final
            if decision == "RAG":
                final_decision = "RAG"
            elif decision == "WEB":
                 final_decision = "WEB"
            else: # Fallback para WEB se a resposta não for exatamente RAG ou WEB
                logfire.warn("Decisão inesperada do roteador, usando WEB como fallback.", raw_decision = raw_decision, query = query, decision_parsed = decision)
                final_decision = "WEB"

            logfire.info("Decisão de roteamento finalizada", raw_decision = raw_decision, final_decision = final_decision)

            return {"source_decision": final_decision}

        except Exception as e:
            logfire.error("Erro no nó de roteamento, usando WEB como fallback.", query = query, error = str(e), exc_info = True)
            return {"source_decision": "WEB"}

# Função para o nó de retrieve do RAG
# Aplica o decorador de rastreamento do LangSmith, definindo tipo de execução e nome do nó
@traceable(run_type = "retriever", name = "Node_RetrieveRAG") # LangSmith Decorator
# Define a função que recebe o estado do grafo e retorna um dicionário com o contexto RAG
def dsa_retrieve_rag_node(state: GraphState) -> dict:
    
    # Extrai a consulta armazenada no estado do grafo
    query = state["query"]
    
    # Inicia um span no LogFire para rastrear a execução deste nó, incluindo a consulta
    span = logfire.span("Executando Nó: Retrieve RAG", query = query)
    
    # Abre o contexto do span para agrupar todas as operações dentro deste nó
    with span:
        
        # Inicia um bloco try para capturar e tratar exceções durante o retrieve
        try:
            
            # Carrega o retriever RAG a partir da função cacheada
            local_retriever = dsa_carrega_retriever()
            
            # Executa a consulta usando o retriever carregado
            results = local_retriever.invoke(query)
            
            # Concatena o conteúdo de cada documento retornado em uma única string de contexto
            context = "\n\n".join([doc.page_content for doc in results])

            # Verifica se não foi encontrado nenhum conteúdo relevante
            if not context:
                
                # Registra no LogFire que nenhum contexto RAG foi localizado
                logfire.info("Nenhum contexto RAG encontrado.")
                
                # Retorna mensagem informando que não há documentos internos relevantes
                return {"rag_context": "Não foram encontrados documentos internos relevantes."}
            
            # Caso o contexto recuperado não esteja vazio
            else:
                
                # Registra no LogFire que o contexto foi encontrado e informa o tamanho do texto
                logfire.info("Contexto RAG encontrado.", context_length = len(context))
                
                # Comentário opcional para depuração: registrar um trecho do contexto (cuidado com dados sensíveis)
                # logfire.debug("Trecho do contexto RAG:", context_snippet=context[:200])
                # Retorna o contexto completo recuperado
                return {"rag_context": context}
        
        # Captura qualquer exceção ocorrida durante o processo de retrieve
        except Exception as e:
            
            # Registra o erro no LogFire, incluindo a consulta, mensagem e traceback
            logfire.error("Erro no nó RAG", query = query, error = str(e), exc_info = True)
            
            # Retorna um dicionário indicando erro ao buscar nos documentos internos
            return {"rag_context": f"Erro ao buscar nos documentos internos: {e}"}

# Função para o nó de busca na web
# Aplica o decorador de rastreamento do LangSmith, definindo tipo de execução e nome do nó
@traceable(run_type = "tool", name = "Node_SearchWeb") # LangSmith Decorator
# Define a função que recebe o estado do grafo e retorna um dicionário com os resultados da web
def dsa_search_web_node(state: GraphState) -> dict:
    
    # Extrai a consulta armazenada no estado do grafo
    query = state["query"]
    
    # Inicia um span no LogFire para rastrear a execução deste nó, incluindo a consulta
    span = logfire.span("Executando Nó: Search Web", query = query)
    
    # Abre o contexto do span para agrupar todas as operações dentro deste nó
    with span:
        
        # Inicia um bloco try para capturar e tratar exceções durante a busca na web
        try:
            
            # Cria a instância da ferramenta de busca DuckDuckGo
            web_search_tool = DuckDuckGoSearchRun()
            
            # Executa a consulta usando a ferramenta de busca
            results = web_search_tool.run(query)

            # Verifica se não houve resultados retornados
            if not results:
                
                # Registra no LogFire que nenhum resultado foi encontrado
                logfire.info("Nenhum resultado da busca web.")
                
                # Retorna mensagem informando ausência de resultados relevantes
                return {"web_results": "Não foram encontrados resultados relevantes na web."}
            
            # Caso existam resultados
            else:
                
                # Registra no LogFire que resultados foram encontrados, incluindo a quantidade
                logfire.info("Resultados da busca web encontrados.", results_length = len(results))
                
                # Comentário opcional para depuração: registrar trecho dos resultados (cuidado com tamanho)
                # logfire.debug("Trecho resultados web:", results_snippet=results[:200])
                # Retorna os resultados completos da busca
                return {"web_results": results}
        
        # Captura qualquer exceção ocorrida durante o processo de busca na web
        except Exception as e:
            
            # Registra o erro no LogFire com detalhes da exceção e traceback
            logfire.error("Erro no nó Web Search", query = query, error = str(e), exc_info = True)
            
            # Retorna um dicionário indicando erro ao realizar a busca
            return {"web_results": f"Erro ao realizar busca na web: {e}"}

# Função para gerar a resposta final para o usuário
@traceable(run_type = "llm", name = "Node_GenerateAnswer") # LangSmith Decorator
def dsa_generate_answer_node(state: GraphState) -> dict:
    
    # Extrai a query do estado
    query = state["query"]

    # Span do LogFire
    span = logfire.span("Executando Nó: Geração da Resposta", query = query)
    
    # Contexto do Span
    with span:

        # Atributos do estado
        rag_context = state.get("rag_context")
        web_results = state.get("web_results")
        context_provided = ""
        source_used = "Nenhuma"

        # Determina se contexto RAG e WEB estão presentes
        rag_useful = rag_context != "Não foram encontrados documentos internos relevantes."
        web_useful = web_results != "Não foram encontrados resultados relevantes na web."

        # Verifica a condição e define os atributos
        if rag_useful:
            context_provided = f"Contexto dos documentos internos:\n{rag_context}"
            source_used = "RAG"
            logfire.info("Usando contexto RAG para gerar resposta.")
        elif web_useful:
            context_provided = f"Resultados da busca na web:\n{web_results}"
            source_used = "WEB"
            logfire.info("Usando resultados da web para gerar resposta.")
        else:
            context_provided = "Nenhuma informação adicional encontrada nas fontes disponíveis."
            logfire.info("Nenhum contexto útil encontrado para gerar resposta.")

        # Log da fonte utilizada
        logfire.info('Fonte(s) para geração', source_used = source_used, rag_context_present = rag_useful, web_results_present = web_useful)

        prompt = f"""Você é um assistente de suporte técnico prestativo e conciso. Responda à pergunta do usuário de forma clara, utilizando APENAS as informações fornecidas no contexto abaixo. Se o contexto não for útil ou relevante para a pergunta, diga que não encontrou informações específicas sobre isso nas fontes disponíveis. NÃO invente respostas.

        Consulta do Usuário: {query}

        Contexto Fornecido:
        {context_provided}

        Resposta Concisa:"""

        # Bloco de execução
        try:

            # Executa a função
            llm_resposta_final = dsa_carrega_llm_resposta_final()
            
            # Executa o LLM
            response = llm_resposta_final.invoke(prompt)

            # Extrai o conteúdo da resposta
            final_answer = response.content

            # LogFire
            logfire.info("Resposta final gerada", source_used = source_used, answer_length = len(final_answer))
            
            return {"final_answer": final_answer}
       
        except Exception as e:
            logfire.error("Erro no nó de geração da resposta", query = query, source_used = source_used, error = str(e), exc_info = True)
            return {"final_answer": f"Desculpe, ocorreu um erro técnico ao tentar gerar a resposta final: {e}"}

# Função para nó de decisão da fonte (RAG ou WEB)
# Não precisa de @traceable pois é lógica simples, mas podemos logar
def dsa_decide_source_edge(state: GraphState) -> Literal["retrieve_rag_node", "search_web_node"]:
    decision = state["source_decision"]
    logfire.debug("Aresta Condicional: Decidindo próximo nó", current_decision = decision) # LogFire nível debug

    if decision == "RAG":
        return "retrieve_rag_node"
    else: # Inclui "WEB" e qualquer fallback
        return "search_web_node"

########## Função Para Compilar o Grafo e Definir Regra de Roteamento ##########

# Função para compilar (criar) o grafo do LangGraph
@st.cache_resource # Cacheia o grafo compilado
def dsa_compile_graph():
    
    # Span
    span = logfire.span("Compilando o grafo LangGraph")
    
    # Contexto do Span
    with span:
        
        print("DSA Log - Compilando o grafo LangGraph...") # Mantém o print para feedback no console durante o cache
        
        try:
            
            # Cria os nós
            graph_builder = StateGraph(GraphState)
            graph_builder.add_node("route_query_node", dsa_route_query_node)
            graph_builder.add_node("retrieve_rag_node", dsa_retrieve_rag_node)
            graph_builder.add_node("search_web_node", dsa_search_web_node)
            graph_builder.add_node("generate_answer_node", dsa_generate_answer_node)
            graph_builder.set_entry_point("route_query_node")

            # Cria a condição para o roteamento
            graph_builder.add_conditional_edges("route_query_node", dsa_decide_source_edge, {
                "retrieve_rag_node": "retrieve_rag_node",
                "search_web_node": "search_web_node",
            })

            # Adiciona as arestas sequenciais
            graph_builder.add_edge("retrieve_rag_node", "generate_answer_node")
            graph_builder.add_edge("search_web_node", "generate_answer_node")
            graph_builder.add_edge("generate_answer_node", END) # Termina o grafo após a geração

            # Compila o grafo
            app = graph_builder.compile()
            print("DSA Log - Grafo compilado com sucesso!")
            
            logfire.info("Grafo LangGraph compilado com sucesso")
            
            return app
        
        except Exception as e:
            
            print(f"DSA Log - Erro GRAVE ao compilar o grafo: {e}") # Print para erro crítico
            
            logfire.critical("Erro ao compilar o grafo LangGraph", error = str(e), exc_info = True)
            
            # Levanta o erro para o Streamlit tratar (que já faz st.error e st.stop)
            raise e

########## Configuração do Streamlit ##########

# Títulos da app
st.title("Data Science Academy - Projeto 6")
st.title("🤖 Assistente de Suporte Técnico Interativo") 

# Inicializa o histórico de chat no session_state se não existir
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar com seu suporte técnico hoje?"}]

# Exibe mensagens do histórico a cada rerun
for message in st.session_state.messages:
    
    # Inicia um bloco de mensagem de chat no Streamlit usando o papel (role) da mensagem
    with st.chat_message(message["role"]):
        
        # Renderiza o conteúdo da mensagem como markdown no chat
        st.markdown(message["content"])
        
        # Verifica se a mensagem foi gerada pelo assistente e inclui informação de fonte
        if message["role"] == "assistant" and "source" in message:
            
            # Exibe a fonte consultada como uma legenda abaixo da mensagem do assistente
            st.caption(f"Fonte consultada: {message['source']}")

# Carrega LLM, Retriever e compila o grafo (usando cache)
llm_final = dsa_carrega_llm_resposta_final()
retriever_rag = dsa_carrega_retriever()
compiled_app = dsa_compile_graph()

# Input do usuário via chat_input
if user_query := st.chat_input("Sua pergunta sobre suporte técnico..."):
    
    # Cria um span no LogFire para rastrear toda a requisição, armazenando a consulta do usuário
    span_chat = logfire.span("Processando consulta do usuário via Chat Interface", query = user_query) # Span para toda a requisição
    
    # Abre o contexto do span para agrupar as operações de processamento da consulta
    with span_chat:

        # Registra no LogFire que uma nova consulta foi recebida via chat_input
        logfire.info("Recebida nova consulta do chat input.")

        # Adiciona a mensagem do usuário ao histórico de chat na sessão e a exibe
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Exibe no chat a mensagem do usuário com o papel "user"
        with st.chat_message("user"):
            st.markdown(user_query)

        # Inicia o bloco para exibir a resposta do assistente no chat
        with st.chat_message("assistant"):
            
            # Usa st.status para fornecer feedback de carregamento de forma detalhada
            with st.status("Pensando... 🧠", expanded = False) as status:
                
                # Inicia o bloco try para capturar erros no processamento do agente
                try:
                    
                    # Exibe texto indicando análise da pergunta e decisão sobre a fonte
                    st.write("Analisando sua pergunta e decidindo a melhor fonte...")
                    
                    # Prepara o dicionário de inputs para invocar o grafo de agentes
                    inputs = {"query": user_query}

                    # Executa o grafo LangGraph compilado com os inputs fornecidos
                    final_state = compiled_app.invoke(inputs)

                    # Atualiza o status indicando qual fonte está sendo consultada
                    st.write(f"Consultando {final_state.get('source_decision', 'fonte desconhecida')}...") # Atualiza o status

                    # Extrai a resposta final do estado retornado pelo grafo
                    final_answer = final_state.get("final_answer", "Não foi possível gerar uma resposta.")
                    
                    # Extrai a decisão da fonte usada
                    source = final_state.get('source_decision', 'N/A')

                    # Atualiza o status para completo, indicando que a resposta está pronta
                    status.update(label = "Resposta pronta!", state = "complete", expanded = False) # Atualiza status para completo

                # Em caso de erro na invocação do grafo, captura a exceção
                except Exception as e:
                    
                    # Registra no LogFire o erro ocorrido, incluindo a consulta e o traceback
                    logfire.error("Erro ao invocar o grafo LangGraph principal a partir do Chat Interface", query = user_query, error = str(e), exc_info = True)
                    
                    # Exibe mensagem de erro na interface do usuário com detalhes da exceção
                    st.error(f"Ocorreu um erro inesperado. Detalhe: {e}")
                    
                    # Define a resposta exibida como mensagem de erro
                    final_answer = f"Desculpe, ocorreu um erro técnico: {e}"
                    
                    # Define a fonte como "Erro" para registro
                    source = "Erro"
                    
                    # Atualiza o status indicando erro no processamento
                    status.update(label = "Erro no processamento", state = "error", expanded = True)

            # Exibe a resposta final gerada pelo assistente no chat
            st.markdown(final_answer)

            # Adiciona a resposta do assistente ao histórico de chat, incluindo a fonte consultada
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "source": source # Armazena a fonte junto com a mensagem
            })

# Instruções podem ir para um expander ou modal em vez da sidebar para um look mais limpo
with st.expander("ℹ️ Instruções e Observações"):
    st.write("""
        - Faça perguntas específicas sobre sua dúvida técnica.
        - O sistema decidirá automaticamente a melhor fonte (documentos internos ou web).
        - IA Generativa pode cometer erros. SEMPRE valide informações críticas.
        - Logs e Traces são enviados para **Pydantic LogFire** e **LangSmith**.
    """)
    if st.button("Suporte DSA"):
        st.write("Dúvidas? Envie um e-mail para: suporte@datascienceacademy.com.br")

