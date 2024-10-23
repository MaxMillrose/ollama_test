from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from tqdm import tqdm
import sys, os

ollama_embbed_model = "mxbai-embed-large:latest"

ollama_query_model = "wizard-vicuna-uncensored:30b"
pdf_path="./"
chroma_path="./vectorstore"
OEmbed = OllamaEmbeddings(model=ollama_embbed_model)
text_splitter = SemanticChunker(
    embeddings=OEmbed,
    breakpoint_threshold_type="gradient", 
    breakpoint_threshold_amount=50.0)


def check_chroma_db(chroma_db):
    if os.path.exists(chroma_db):
        vectorstore = Chroma(embedding_function=OEmbed, persist_directory=chroma_db)
        print(f"Existing vector store found with {vectorstore._collection.count()} documents.")
        return chroma_db
    else:
        return False
# End of check_chromadb


def load_pdfs(pdf_path):
    loader = DirectoryLoader(path=pdf_path, loader_cls=PyPDFLoader, glob="*.pdf")   
    docs = loader.load()  # Load all PDF files
    
    if not docs:
        print("No documents loaded from the specified PDFs.")
        sys.exit(1)  # Exit if no documents are loaded
    else:
        print(f"Number of PDF files loaded: {len(docs)}")
        return docs
    


def splitting(docs):
    chunks = []
    for doc in tqdm(docs, desc="Splitting Documents"):
        doc_chunks = text_splitter.split_documents([doc])  # Split one document at a time
        chunks.extend(doc_chunks)  # Add the chunks to the list
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OEmbed,persist_directory=chroma_path)
    return vectorstore

def questions(vectorstore):
    Question_Ling="Please tell me about Ling."
    sim_search = vectorstore.similarity_search(Question_Ling)
    print(f"Our sim_search is {len(sim_search)}.")




def main():
    print(f"ONA - Ollama Novel Assistant")
    
    if not check_chroma_db(chroma_path):
        print(f"Found no existing vectorstore. ")
        vectorstore = splitting(load_pdfs(pdf_path))
        questions(vectorstore)

    
    sys.exit(1)



if __name__ == '__main__' :
    main()
