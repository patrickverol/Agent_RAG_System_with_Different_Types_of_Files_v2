# Projeto 3 - Re-Ranking, Agentic RAG com LangGraph e LM Studio Para Assistente de Processo de Licitação

# Imports
import os
import numpy as np
import langgraph
import streamlit as st
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity

# Evita problema de compatibilidade entre Streamlit e PyTorch
import torch
torch.classes.__path__ = []

# Filtra warnings
import warnings
warnings.filterwarnings('ignore')

# Desativa paralelismo de tokenização para evitar problemas de compatibilidade com diferentes SOs
os.environ["TOKENIZERS_PARALLELISM"] = "false"


########## Etapa 1 - Configuração da Interface do Streamlit ##########


# Configuração da página do Streamlit
st.set_page_config(page_title="Data Science Academy", page_icon=":100:", layout="centered")

# Barra lateral com instruções
st.sidebar.title("Instruções")
st.sidebar.write("""
- Digite perguntas específicas sobre o processo de licitação para obter respostas detalhadas.
- O assistente de IA vai utilizar a base de dados do RAG para gerar respostas customizadas.
- Documentos, contratos e procedimentos complementares podem ser usados para aperfeiçoar o sistema de RAG (que nesse caso deve ser recriado com cada novo documento).
- IA Generativa comete erros. SEMPRE valide as respostas.
""")

# Botão de suporte na barra lateral
if st.sidebar.button("Suporte"):
    st.sidebar.write("Dúvidas? Envie um e-mail para: suporte@datascienceacademy.com.br")

# Títulos principais
st.title("DSA - Projeto 3")
st.title("Re-Ranking, Agentic RAG com LangGraph e LM Studio Para Assistente de Processo de Licitação")


########## Etapa 2 - Configuração do Modelo LLM e do Processo de Retrieve do RAG ##########


# Inicializa o modelo de linguagem (LLM)
llm = ChatOpenAI(model_name = "hermes-3-llama-3.2-3b@q6_k",
                 openai_api_base = "http://10.0.0.20:1234/v1",
                 openai_api_key = "lm-studio",  # Chave fictícia para uso com LM Studio
                 temperature = 0.3,
                 max_tokens = 256)

# Modelo de embeddings para busca semântica (deve ser o mesmo usado no RAG)
embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-base-en")

# Banco de dados vetorial para recuperação de informações
vector_db = Chroma(persist_directory = "rag/chroma_db", embedding_function = embedding_model)

# Configuração do retriever para busca de documentos
retriever = vector_db.as_retriever()

# Template de prompt para interação com o modelo
prompt = PromptTemplate.from_template(
    "Você é um assistente especializado em licitação de órgãos públicos. Responda à seguinte pergunta, em português do Brasil, com base nos documentos fornecidos:\n{context}\nPergunta: {input}"
)

# Pipeline de processamento de consulta
# O uso do RunnablePassthrough é indicado sempre que você precisa encaminhar a entrada inicial do pipeline para etapas posteriores sem alterações, 
# permitindo maior flexibilidade ao construir prompts compostos ou sofisticados.
combine_docs_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

# Criação da cadeia de recuperação
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)


########## Etapa 3 - Configuração de Ferramentas Para o Agente de IA ##########


# Wrapper de busca usando DuckDuckGo
search = DuckDuckGoSearchAPIWrapper(region = "br-pt", max_results = 5)

# Ferramenta de busca baseada na Web
web_search_tool = Tool(name = "WebSearch",
                       func = search.run,
                       description = "Busca informações atualizadas na web sobre licitação pública ou assuntos relacionados.")

# Inicializa o agente com ferramenta de busca na web
dsa_agent_executor = initialize_agent(tools = [web_search_tool],
                                      llm = llm,
                                      agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                      verbose = True,
                                      handle_parsing_errors = True)


########## Etapa 4 - Configuração dos Componentes do Agente de IA ##########


# Definição do estado do agente
class AgentState(BaseModel):
    query: str
    next_step: str = ""
    retrieved_info: list = []
    possible_responses: list = []
    similarity_scores: list = []
    ranked_response: str = ""
    confidence_score: float = 0.0

# Definição dos passos de decisão do agente
def dsa_passo_decisao_agente(state: AgentState) -> AgentState:
    
    # Converte a consulta do usuário para letras minúsculas para padronizar a análise
    query = state.query.lower()

    # Se detectar palavras-chave que pedem explicação ou conceito geral, define o próximo passo como geração direta de resposta (sem usar RAG)
    if any(palavra in query for palavra in ["explique", "resuma", "defina", "conceito", "geral", "o que é"]):
        state.next_step = "gerar"

    # Caso detecte pedido explícito de busca atualizada na web, define o próximo passo como uso da ferramenta de busca na web (sem usar RAG)
    elif any(palavra in query for palavra in ["busque na web", "notícias", "atualizado", "recente", "últimas informações"]):
        state.next_step = "usar_web"

    # Se nenhuma das condições anteriores for atendida, usa o método padrão RAG com recuperação de documentos
    else:
        state.next_step = "retrieve"

    # Retorna o estado atualizado com a decisão tomada
    return state

# Função que aciona o agente com ferramenta de busca na web
def dsa_usar_ferramenta_web(state: AgentState) -> AgentState:
    
    # Executa a ferramenta de busca na web usando o agente com a consulta do usuário
    resultado = dsa_agent_executor.invoke(state.query)
    
    # Extrai a resposta obtida pela ferramenta ou retorna mensagem padrão se não houver resposta válida
    state.ranked_response = resultado.get("output", "Nenhuma informação obtida pela busca web.")
    
    # Define um score de confiança fixo para a resposta obtida via busca web
    state.confidence_score = 0.0  
    
    # Retorna o estado atualizado após a execução da busca
    return state

# Recuperação de documentos
def dsa_retrieve_info(state: AgentState) -> AgentState:

    # Obtém documentos relevantes a partir da consulta atual
    retrieved_docs = retriever.invoke(state.query)
    
    # Armazena os documentos recuperados no estado do agente
    state.retrieved_info = retrieved_docs
    
    # Retorna o estado atualizado
    return state

# Geração de múltiplas respostas
def dsa_gera_multiplas_respostas(state: AgentState) -> AgentState:

    # Gera uma lista com 5 respostas diferentes para a mesma consulta
    responses = [qa_chain.invoke({"input": state.query}) for _ in range(5)]
    
    # Armazena as respostas geradas no estado do agente
    state.possible_responses = responses
    
    # Retorna o estado atualizado
    return state

# Avaliação de similaridade entre respostas e documentos (Passo 1 do Re-Ranking)
def dsa_avalia_similaridade(state: AgentState) -> AgentState:

    # Extrai o conteúdo textual dos documentos recuperados
    retrieved_texts = [doc.page_content for doc in state.retrieved_info]
    
    # Recupera as respostas geradas anteriormente
    responses = state.possible_responses

    # Gera embeddings para os textos dos documentos, se houver
    retrieved_embeddings = embedding_model.embed_documents(retrieved_texts) if retrieved_texts else []
    
    # Extrai os textos das respostas, tratando diferentes formatos
    response_texts = [response["answer"] if isinstance(response, dict) and "answer" in response else str(response) for response in responses]
    
    # Gera embeddings para os textos das respostas, se houver
    response_embeddings = embedding_model.embed_documents(response_texts) if response_texts else []

    # Se não houver embeddings válidos, preenche com zeros e retorna
    if not retrieved_embeddings or not response_embeddings:
        state.similarity_scores = [0.0] * len(response_texts)
        return state

    # Calcula a similaridade média entre cada resposta e os documentos recuperados
    similarities = [
        np.mean([cosine_similarity([response_embedding], [doc_embedding])[0][0] for doc_embedding in retrieved_embeddings])
        for response_embedding in response_embeddings
    ]

    # Armazena as pontuações de similaridade no estado do agente
    state.similarity_scores = similarities
    
    # Retorna o estado atualizado
    return state

# Ranqueamento das respostas geradas (Passo 2 do Re-Ranking)
def dsa_rank_respostas(state: AgentState) -> AgentState:

    # Combina cada resposta gerada com sua pontuação de similaridade correspondente
    response_with_scores = list(zip(state.possible_responses, state.similarity_scores))
    
    # Verifica se existem respostas com pontuações para ranquear
    if response_with_scores:
        
        # Ordena as respostas com base na similaridade, da maior para menor
        ranked_responses = sorted(response_with_scores, key=lambda x: x[1], reverse=True)
        
        # Seleciona a melhor resposta (com maior similaridade)
        state.ranked_response = ranked_responses[0][0]
        
        # Salva o score da melhor resposta como pontuação de confiança
        state.confidence_score = ranked_responses[0][1]

    else:
        
        # Define uma mensagem padrão caso não existam respostas válidas
        state.ranked_response = "Desculpe, não encontrei informações relevantes para esta consulta."
        
        # Define a confiança como 0, pois nenhuma resposta válida foi encontrada
        state.confidence_score = 0.0
    
    # Retorna o estado atualizado após ranqueamento
    return state


########## Etapa 5 - Configuração do Fluxo de Execução do Agente com LangGraph ##########


# Inicializa o grafo de estados para controlar o fluxo do agente
workflow = StateGraph(AgentState)

# Adiciona o nó de decisão, responsável por definir o próximo passo com base na consulta do usuário
workflow.add_node("decision", dsa_passo_decisao_agente)

# Adiciona o nó responsável pela recuperação de documentos relevantes (RAG)
workflow.add_node("retrieve", dsa_retrieve_info)

# Adiciona o nó para geração de múltiplas respostas usando o modelo LLM
workflow.add_node("generate_multiple", dsa_gera_multiplas_respostas)

# Adiciona o nó para avaliar similaridade entre respostas geradas e documentos recuperados
workflow.add_node("evaluate_similarity", dsa_avalia_similaridade)

# Adiciona o nó que ranqueia as respostas com base nas similaridades avaliadas
workflow.add_node("rank_responses", dsa_rank_respostas)

# Adiciona o nó que utiliza a ferramenta de busca na web para consultas atualizadas
workflow.add_node("usar_web", dsa_usar_ferramenta_web)

# Define o ponto inicial do fluxo como sendo o nó de decisão
workflow.set_entry_point("decision")

# Define transições condicionais partindo do nó de decisão, escolhendo o próximo nó baseado no estado
workflow.add_conditional_edges(
    "decision",
    lambda state: {
        "retrieve": "retrieve",
        "gerar": "generate_multiple",
        "usar_web": "usar_web"
    }[state.next_step]
)

# Define que após a recuperação de documentos, o fluxo sempre gera múltiplas respostas
workflow.add_edge("retrieve", "generate_multiple")

# Define que após gerar múltiplas respostas, avalia-se a similaridade com documentos
workflow.add_edge("generate_multiple", "evaluate_similarity")

# Define que após a avaliação de similaridade, ocorre o ranqueamento das respostas
workflow.add_edge("evaluate_similarity", "rank_responses")

# Compila o fluxo completo para ser executado pelo agente
agent_workflow = workflow.compile()


########## Etapa 6 - Configuração do Fluxo da Web App no Streamlit ##########


# Campo de entrada para o usuário digitar sua pergunta
query = st.text_input("Digite sua pergunta:")

# Quando o botão "Enviar" for pressionado
if st.button("Enviar"):

    # Exibe spinner enquanto o agente está processando a consulta
    with st.spinner("O Sistema de IA Está Processando Sua Consulta. Pratique a Paciência e Aguarde..."):
        
        # Executa o fluxo do agente com o estado inicial contendo a consulta
        output = agent_workflow.invoke(AgentState(query=query))
    
    # Exibe o título da resposta ranqueada
    st.subheader("Resposta:")
    
    # Extrai a resposta ranqueada do resultado do agente
    resposta = output.get("ranked_response", "Nenhuma resposta gerada.")
    
    # Extrai o score de confiança associado à resposta
    confidence = output.get("confidence_score", 0.0)

    # Caso a resposta seja um dicionário, extrai apenas o campo 'answer'
    if isinstance(resposta, dict) and "answer" in resposta:
        resposta = resposta["answer"]
    
    # Mostra a resposta ao usuário usando markdown para formatação
    st.markdown(resposta)

    # Exibe o título para a pontuação de confiança da resposta
    st.subheader("Confiança da Resposta com Base no RAG:")
    
    # Apresenta o Confidence Score com formatação especial
    st.markdown(f"`{confidence:.2f}`")

    # Recupera documentos relacionados a partir da saída do agente
    documentos_relacionados = output.get("retrieved_info", [])

    # Verifica se existem documentos relacionados retornados
    if documentos_relacionados:
        
        # Exibe o título da seção dos documentos relacionados
        st.subheader("Documentos Relacionados:")
        
        # Itera sobre cada documento retornado para apresentá-los ao usuário
        for doc in documentos_relacionados:
            
            # Mostra o ID único do documento
            st.markdown(f"**ID:** `{doc.id}`")
            
            # Mostra a fonte original do documento
            st.markdown(f"**Fonte:** `{doc.metadata.get('source', 'Desconhecida')}`")
            
            # Apresenta o conteúdo textual do documento em uma caixa de texto
            st.text_area("Conteúdo", doc.page_content, height = 80)
            
    else:
        # Mensagem exibida caso não haja documentos relacionados encontrados
        st.write("Nenhum documento relacionado encontrado.")

# Rastreabilidade da aplicação
APP_WATERMARK = "DSA-PROJETO3-AGENTICRAG-LANGGRAPH-982310098667743221"

# Exibe marca d'água no rodapé da aplicação
st.markdown(f"<div style='text-align: center; color: #cccccc; font-size:10px;'>{APP_WATERMARK}</div>", unsafe_allow_html=True)


