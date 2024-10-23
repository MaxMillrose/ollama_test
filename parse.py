import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_ollama import OllamaEmbeddings  # Updated import
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
import sys
from tqdm import tqdm  # Import tqdm for progress bars

# Variables for loader and LLM
ollama_model = "wizard-vicuna-uncensored:30b" 
source_path = "./"  # Current directory for PDF files
persist_dir = "./chroma_db"  # Directory to persist vector store

# Initialize embeddings
ollama_embed = OllamaEmbeddings(model=ollama_model)

# Load documents from the specified directory
try:
    print("Loading PDF documents...")
    # Create a loader for the specific PDF files
    loader = DirectoryLoader(path=source_path, loader_cls=PyPDFLoader, glob="*.pdf")
    
    # Load all PDF documents
    doc_data = loader.load()  # Load all PDF files

    if not doc_data:
        print("No documents loaded from the specified PDFs.")
        sys.exit(1)  # Exit if no documents are loaded
    
    # Extract PDF file names from loaded documents
    pdf_files = [doc.metadata.get('source', 'Unknown') for doc in doc_data]
    print(f"Loaded {len(doc_data)} documents from the following PDFs:")
    print(pdf_files)  # Print the list of found PDF files
except Exception as e:
    print(f"Error loading documents: {e}")
    sys.exit(1)  # Exit if document loading fails

# Initialize the text splitter
text_splitter = SemanticChunker(
    embeddings=ollama_embed, 
    buffer_size=5, 
    breakpoint_threshold_amount=50.0  # Adjust as needed
)

# Split documents into chunks
print("Chunking documents...")
docs = []
chunked_files = []  # List to store names of processed files

# Use tqdm to show progress for chunking documents
for doc in tqdm(doc_data, desc="Chunking Documents", total=len(doc_data)):
    chunked_docs = text_splitter.split_documents([doc])
    docs.extend(chunked_docs)
    chunked_files.append(doc.metadata.get('source', 'Unknown'))  # Assuming metadata contains the source file name

    # Show progress for each chunk
    for chunk in tqdm(chunked_docs, desc="Processing Chunks", leave=False):
        # Optionally perform any operations on each chunk here
        pass  # This is just to show progress; remove or modify as needed

print(f"Split into {len(docs)} chunks from the following files:")
print(chunked_files)  # Print the list of processed files

# Check if any chunks were created before creating the vector store
if not docs:
    print("No chunks were created, exiting.")
    sys.exit(1)

# Create a Chroma vector store from the documents
print("Creating Chroma vector store...")
vectordb = Chroma(persist_directory=persist_dir, embedding=ollama_embed)  # Initialize without documents

# Persist the vector database with progress tracking
print("Persisting the vector database...")
for doc in tqdm(docs, desc="Persisting Chunks", total=len(docs)):
    vectordb.add_documents([doc])  # Add each chunk to the vector store

# Save the vector store
vectordb.persist()  # This typically saves the vector store to disk
print("Vector database has been persisted successfully.")
