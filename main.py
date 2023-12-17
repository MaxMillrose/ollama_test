#from langchain.document_loaders import OnlinePDFLoader
#from langchain.document_loaders import UnstructuredPDFLoader
#from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain import prompts
import sys
import os

# VARS for loader and LLM
# PROD model
#ollama_model = "wizard-vicuna-uncensored:30b" 
ollama_model = "llama2"
source_path = "./sources"
persist_dir = "./chroma_db"
loader_class="PyMuPDFLoader"
max_concurrency=4
#gpt_embed = GPT4AllEmbeddings()
ollama_embed=OllamaEmbeddings(base_url="http://localhost:11434", 
                            model=ollama_model,
                            num_thread=4, show_progress=True )

llm = Ollama(base_url='http://localhost:11434',model=ollama_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
print(llm("Please respond with a single dot"))

def load_data():

    # load the pdf and split it into chunks
    print(f"Using Model from {ollama_model}")        
    print(f"Using persist_directory ./chroma_db")        
    #print(f"max_concurrency is {max_concurrency}")        
    print(f"Loading documents from {source_path}")        

    # Langchian load / parse documents
    print(f"Loading documents from {source_path}")        
    loader = DirectoryLoader(path=source_path, loader_cls=PyPDFLoader)
    doc_data = loader.load()
    print(len(doc_data))


    print(f"Splitting the text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
    docs = text_splitter.split_documents(doc_data)
    vectorstore = Chroma.from_documents(documents=docs, persist_directory=persist_dir, 
                                        embedding=ollama_embed) 
    print(f"Our new collection count is : ")
    print(vectorstore._collection.count())
# end of load_data()


if (os.path.exists(persist_dir)):
    vectorstore = Chroma(persist_directory=persist_dir, 
                        embedding_function=ollama_embed)
    print(f"Our collection count from persistent is : ")
    print(vectorstore._collection.count())
else:
    load_data()



while True:
    query = input("\nQuery: ")
    if (query == "exit" or query == "/bye"):
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = prompts.PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})
