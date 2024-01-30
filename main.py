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
gpt_embed = GPT4AllEmbeddings()
my_embedding=gpt_embed
ollama_embed=OllamaEmbeddings(base_url="http://localhost:11434", 
                            model=ollama_model,
                            num_thread=2, show_progress=True )

llm = Ollama(base_url='http://localhost:11434',model=ollama_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

def new_vectordb():
    print(f"parsing and loading new sources")
    loader = DirectoryLoader(path=source_path, loader_cls=PyPDFLoader)
    doc_data = loader.load()
    print(len(doc_data))

    print(f"Splitting the text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=2500)
    docs = text_splitter.split_documents(doc_data)
    vectordb = Chroma.from_documents(documents=docs, persist_directory=persist_dir, 
                                        embedding=my_embedding) 
    print(f"Our new collection count is : ")
    print(vectordb._collection.count())
    vectordb.persist()

    return vectordb
# end of load_data()


if (os.path.exists(persist_dir)):
    vectordb = Chroma(persist_directory=persist_dir, 
                        embedding_function=my_embedding)
    print(f"Our collection count from persistent is : ")
    print(vectordb._collection.count())
else:
    vectordb = new_vectordb()


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
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

while True:
    query = input("\nQuery: ")
    if (query == "exit" or query == "/bye"):
        break
    if query.strip() == "":
        continue
    
    result = qa_chain({"query": query})
