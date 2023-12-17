# filename: parse.py
# parse the documents, split and embed into model 
# with persistent dir

#from langchain.document_loaders import OnlinePDFLoader
#from langchain.document_loaders import UnstructuredPDFLoader
#from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

# VARS for loader and LLM
# PROD model
#ollama_model = "wizard-vicuna-uncensored:30b" 
ollama_model = "llama2"
source_path = "./sources"
#max_concurrency=4

# load the pdf and split it into chunks
print(f"Using persist_directory ./chroma_db ")        
# print(f"max_concurrency is {max_concurrency}")        
print(f"Loading documents from {source_path}")        



# Langchian load / parse documents
loader = DirectoryLoader(path=source_path, loader_cls=PyPDFLoader)
doc_data = loader.load()
print(len(doc_data))

#for document in doc_data:
#    print(str(document.metadata["page"]) + ":", document.page_content[:300])


print(f"Splitting the text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
docs = text_splitter.split_documents(doc_data)
 
embed=OllamaEmbeddings(base_url="http://localhost:11434", 
                        model=ollama_model,
                        num_thread=4, show_progress=True )

vectorstore = Chroma.from_documents(documents=docs, persist_directory="./chroma_db", 
                                    embedding=embed)

