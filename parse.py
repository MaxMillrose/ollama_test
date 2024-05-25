from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain import prompts
import sys
import os

# VARS for loader and LLM
ollama_model = "wizard-vicuna-uncensored:30b" 
#ollama_model = "llama2"
source_path = "/tmp/llama/sources"
persist_dir = "./chroma_db"
loader_class="PyMuPDFLoader"
max_concurrency=4
#gpt_embed = GPT4AllEmbeddings()
ollama_embed=OllamaEmbeddings(model=ollama_model, show_progress=True, num_thread=9, num_ctx=4096 )


my_embedding=ollama_embed

llm = Ollama(model=ollama_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

print(f"parsing and loading new sources")
loader = DirectoryLoader(path=source_path, loader_cls=PyPDFLoader)
doc_data = loader.load()
print(len(doc_data))

print(f"Splitting the text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
docs = text_splitter.split_documents(doc_data)
vectordb = Chroma.from_documents(documents=docs, persist_directory=persist_dir, 
                                    embedding=my_embedding) 
print(f"Our new collection count is : ")
print(vectordb._collection.count())
vectordb.persist()

