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
# PROD model
#ollama_model = "wizard-vicuna-uncensored:30b" 
ollama_model = "llama2"
source_path = "./sources"
persist_dir = "./chroma_db"
max_concurrency=4
#gpt_embed = GPT4AllEmbeddings()
ollama_embed=OllamaEmbeddings(base_url="http://localhost:11434", 
                            model=ollama_model,
                            num_thread=4, show_progress=True )

my_embedding=ollama_embed
llm = Ollama(base_url='http://localhost:11434',model=ollama_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

if (os.path.exists(persist_dir)):
    vectordb = Chroma(persist_directory=persist_dir, 
                        embedding_function=my_embedding)
    print(f"Our collection count from persistent is : ")
    print(vectordb._collection.count())
else:
        print(f"Could not find ./chroma_db  ")
        print(f"Please run parse.py first ")
        exit


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
    #query = "Tell me about Patrick"
    query = input("\nQuery: ")
    if (query == "exit" or query == "/bye"):
        break
    if query.strip() == "":
        continue
    
    result = qa_chain({"query": query})
