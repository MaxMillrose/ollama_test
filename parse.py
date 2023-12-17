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

import sys
import os

# VARS for loader and LLM
# PROD model
#ollama_model = "wizard-vicuna-uncensored:30b" 
ollama_model = "llama2"
source_path = "./sources"
persist_directory = "./persist"
loader_class="PyMuPDFLoader"
max_concurrency=4


class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# load the pdf and split it into chunks
print(f"Using persist_directory from {persist_directory}")        
print(f"max_concurrency is {max_concurrency}")        
print(f"Loading documents from {source_path}")        


# Generate MD5-checksum check
# false -> rescan docs from source_path

loader = DirectoryLoader(path=source_path, loader_cls=PyPDFLoader)

doc_data = loader.load_and_split()
print("\n")  
print(len(doc_data))


print(f"Splitting the text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=0)
docs = text_splitter.split_documents(doc_data)
print(f"Init embeding model - data gets send to model")
vectorstore = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(base_url="http://localhost:11434", model=ollama_model))


#with SuppressStdout():
    #vectorstore = Chroma.from_documents(documents=docs, embedding=GPT4AllEmbeddings())   



question="How would describe the protagonists?"
print(f"Using question {question}")
docs = vectorstore.similarity_search(question)

qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
qachain({"query": question})


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

    llm = Ollama(model=ollama_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    result = qa_chain({"query": query})
