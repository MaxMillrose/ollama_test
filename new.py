from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from tqdm import tqdm
import sys, os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain




ollama_embbed_model = "mxbai-embed-large:latest"

ollama_query_model = "wizard-vicuna-uncensored:30b"
pdf_path="./"
chroma_path="./vectorstore"
OEmbed = OllamaEmbeddings(model=ollama_embbed_model)
text_splitter = SemanticChunker(
    embeddings=OEmbed,
    breakpoint_threshold_type="gradient", 
    breakpoint_threshold_amount=50.0)
llm = OllamaLLM(model=ollama_embbed_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0.9)
 
def load_vecstore(chroma_path):
    if os.path.exists(chroma_path):
        vectorstore = Chroma(embedding_function=OEmbed, persist_directory=chroma_path)
        print(f"Existing vector store found with {vectorstore._collection.count()} documents.")
        return vectorstore
    else:
        return False

def load_pdfs(pdf_path):
    loader = DirectoryLoader(path=pdf_path, loader_cls=PyPDFLoader, glob="*.pdf")   
    docs = loader.load()  # Load all PDF files
    
    if not docs:
        print(f"No documents loaded from the specified PDFs.")
        print(f"please check {pdf_path} for PDFs.")
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
    print(f"Asking our questions")
    print(f"NEXT TIME. EXIT for now")
    sys.exit(1)
    
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Please answer the following question and don't make up stuff. {topic}"
        )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    questions_dict = {
    "Question_Ling":"Please tell me about Ling.",
    "Question_Patrick":"Please tell me about Patrick."
    }

    for question in questions_dict: 
        sim_search = vectorstore.similarity_search(question)
        print(f"Our sim_search for {question} is {len(sim_search)}.")
        chain.invoke(question)    
        


def main():
    vectorstore = load_vecstore(chroma_path)
    
    if not vectorstore:
        print(f"Found no existing vectorstore. ")
        print(f"Reparsing documents ")
        vectorstore = splitting(load_pdfs(pdf_path))
        print(vectorstore._collection.get(include=))
    else: 
        print(vectorstore.get)
    
    questions(vectorstore)
    



if __name__ == '__main__' :
    main()
