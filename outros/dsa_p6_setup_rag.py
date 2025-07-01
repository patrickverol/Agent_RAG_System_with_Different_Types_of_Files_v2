# Projeto 6 - Multi-Agentes de IA, RAG, Roteamento, Guardrails, Observabilidade e Explicabilidade com LangGraph, LangSmith e Pydantic LogFire
# Módulo de RAG

# Importa o módulo os para operações do sistema, como verificar existência de diretórios e manipular caminhos
import os

# Importa o carregador de diretórios de PDF para ler múltiplos arquivos PDF do sistema de suporte
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Importa o splitter de texto para dividir documentos em blocos menores para o RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importa FAISS para criação de índices vetoriais a partir de documentos
from langchain_community.vectorstores import FAISS

# Importa o modelo de embeddings FastEmbed para transformar texto em vetores
from langchain_community.embeddings import FastEmbedEmbeddings

# Define o diretório onde os PDFs de suporte técnico devem estar armazenados
DATA_DIR = "dsa_pdfs"

# Define o caminho onde o índice vetorial FAISS será salvo localmente
VECTORSTORE_PATH = "dsa_faiss_index"

# Verifica se o diretório de dados existe; se não, cria e solicita inclusão de PDFs
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Diretório '{DATA_DIR}' criado. Por favor, adicione seus PDFs de suporte técnico nele.")
    exit()

# Função principal para criar o banco de dados vetorial a partir dos PDFs
def dsa_cria_vectordb():
    
    # Exibe mensagem de início de carregamento dos PDFs
    print(f"Carregando PDFs do diretório: {DATA_DIR}")
    try:
        
        # Inicializa o loader para carregar todos os PDFs do diretório
        pdf_loader = PyPDFDirectoryLoader(DATA_DIR, recursive = True)
        documents = pdf_loader.load()
        
        # Se nenhum documento for encontrado, informa e retorna False
        if not documents:
            print(f"Nenhum documento PDF encontrado em '{DATA_DIR}'. Verifique o diretório.")
            return False
            
        print(f"Carregados {len(documents)} páginas/documentos PDF.")

    except Exception as e:
        
        # Captura erros no carregamento e retorna False
        print(f"Erro ao carregar PDFs: {e}")
        return False

    # Informa que a divisão em chunks está começando
    print("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    
    # Realiza a divisão dos documentos em blocos menores
    docs_split = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(docs_split)} chunks.")

    # Inicializa o modelo de embedding FastEmbed para vetorização
    print("Inicializando modelo de embedding (FastEmbed)...")
    embedding_model = FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

    # Cria o índice FAISS na memória e tenta salvá-lo localmente
    print("Criando índice vetorial FAISS...")
    try:
        vector_store = FAISS.from_documents(docs_split, embedding_model)
        print("Índice FAISS criado na memória.")
        vector_store.save_local(VECTORSTORE_PATH)
        print(f"Índice FAISS salvo localmente em: {VECTORSTORE_PATH}")
        return True
    except Exception as e:
        # Captura erros na criação ou salvamento do índice e retorna False
        print(f"Erro ao criar ou salvar o índice FAISS: {e}")
        return False

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    
    # Informa início do processo de configuração do RAG
    print("\nIniciando processo de configuração do RAG...")
    
    # Chama a função de criação do banco vetorial e exibe resultado
    if dsa_cria_vectordb():
        print("Configuração do RAG concluída com sucesso!")
        print(f"O índice vetorial está salvo em '{VECTORSTORE_PATH}'.")
        print(f"Certifique-se de ter seus PDFs na pasta '{DATA_DIR}'.\n")
    else:
        print("\nA configuração do RAG falhou. Verifique os erros acima.")
