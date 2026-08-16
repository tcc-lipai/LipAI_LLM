import os
import re

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PASTA_DADOS = "./Dados"
PASTA_CHROMA = "./chroma_db"


def extrair_fonema_do_caminho(caminho: str) -> str:
    m = re.search(r"fonema_([A-Za-zÀ-ú])", caminho, re.IGNORECASE)
    return m.group(1).upper() if m else "?"


def inicializar_banco_vetorial():
    if not os.path.exists(PASTA_DADOS):
        print(f" Erro: A pasta {PASTA_DADOS} não foi encontrada!")
        return

    print(" Carregando os arquivos Dicas.docx de cada fonema...")

    loader = DirectoryLoader(
        PASTA_DADOS, glob="**/Dicas.docx", loader_cls=Docx2txtLoader, recursive=True
    )
    documentos = loader.load()

    if not documentos:
        print("Nenhum Dicas.docx encontrado dentro de Dados/fonema_X/.")
        return

    for doc in documentos:
        doc.metadata["fonema"] = extrair_fonema_do_caminho(doc.metadata.get("source", ""))

    print(f"{len(documentos)} arquivo(s) de dica carregados. Dividindo em chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Gerados {len(chunks)} blocos de texto.")

    print("Inicializando modelo de embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Gravando dados e indexando no ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PASTA_CHROMA,
    )
    print(f" Banco vetorial de DICAS concluído em: {PASTA_CHROMA}!")


if __name__ == "__main__":
    inicializar_banco_vetorial()