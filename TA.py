# Import required libraries
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import shutil

# Define the LLM model to be used
llm_model = "llama3.1:8b"

# Configure ChromaDB
# Define the path to ChromaDB storage
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")

# Delete the existing ChromaDB storage if it exists (reset on every script run)
if os.path.exists(chroma_db_path):
    shutil.rmtree(chroma_db_path)

# Reinitialize ChromaDB client with a fresh database
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
    )
)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  # Use the custom embedding function,
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    """
    Add documents to the ChromaDB collection.
    
    Args:
        documents (list of str): The documents to add.
        ids (list of str): Unique IDs for the documents.
    """
    collection.add(
        documents=documents,
        ids=ids
    )

# Example: Add sample documents to the collection
documents = [
    "Artificial intelligence is the simulation of human intelligence processes by machines.",
    "Python is a programming language that lets you work quickly and integrate systems more effectively.",
    "ChromaDB is a vector database designed for AI applications.",
    "Fasilkom UI is located in West Jakarta.",
    "Azmi Rahmadisha is the dean of Fasilkom UI.",
    "Fasilkom UI was established in 1945. It is the oldest computer science faculty in Indonesia.",
    "Fasilkom UI is ranked 1st in the world for computer science majors.",
]

doc_ids = ["doc1","doc2","doc3","doc4","doc5","doc6","doc7"]

# Documents only need to be added once or whenever an update is required. 
# This line of code is included for demonstration purposes:
add_documents_to_collection(documents, doc_ids)

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=1):
    """
    Query the ChromaDB collection for relevant documents.
    
    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
    
    Returns:
        list of dict: The top matching documents and their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    """
    Send a query to Ollama and retrieve the response.
    
    Args:
        prompt (str): The input prompt for Ollama.
    
    Returns:
        str: The response from Ollama.
    """
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for Retrieval-Augmented Generation
def rag_pipeline(query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(query_text)
    print("######## RAG PIPELINE ########")
    print(retrieved_docs)
    print(metadata)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Step 2: Send the query along with the context to Ollama
    augmented_prompt = f"""
    You are an expert AI assistant specializing in answering user queries based on the given context or knowledge base (documents).
    Your responses should be clear, concise, and directly related to the context provided.

    Context:
    {context}

    Question:
    {query_text}

    Guidelines:
    - Answer concisely but with enough details.
    - If the context does not contain the answer, state that explicitly.
    - If necessary, provide additional insights based on general knowledge.

    Answer:
    """
    # f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

# Example usage
# Define a query to test the RAG pipeline
query = "Who is the dean of Fasilkom UI?" 
response = rag_pipeline(query)
print("######## Response from LLM ########\n", response)