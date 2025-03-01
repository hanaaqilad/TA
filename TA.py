from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import shutil
import pypdf  

# Define the LLM model to be used
llm_model = "llama3.1:8b"

# Configure ChromaDB
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")

# Delete the existing ChromaDB storage if it exists (reset on every script run)
if os.path.exists(chroma_db_path):
    shutil.rmtree(chroma_db_path)

# Reinitialize ChromaDB 
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self.langchain_embeddings.embed_documents(input)
        
        if isinstance(embeddings, list) and isinstance(embeddings[0], float):
            embeddings = [embeddings]  # Convert single vector into list of lists
        
        return embeddings

# Initialize the embedding function with Ollama embeddings
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(
        model=llm_model,
        base_url="http://localhost:11434"
    )
)

# Define a collection for the RAG workflow
collection_name = "rag_collection_demo"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding  
)

def extract_text_from_pdf(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        text += extracted_text + "\n" if extracted_text else ""  
    return text.strip()

def add_pdf_to_collection(pdf_path, doc_id):
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print(f"⚠️ No text extracted from {pdf_path}. Skipping.")
        return
    
    try:
        embeddings = embedding([text])
        if not embeddings:
            print(f"⚠️ Embeddings failed for {doc_id}. Skipping.")
            return
    except Exception as e:
        print(f"❌ Error generating embeddings for {doc_id}: {e}")
        return

    collection.add(documents=[text], ids=[doc_id])
    print(f"✅ PDF '{doc_id}' added to ChromaDB.")
    
    all_docs = collection.get()
    print("📌 Total documents in collection:", len(all_docs["ids"]))
    print("📝 Stored document IDs:", all_docs["ids"])

def process_pdf_file(pdf_path):
    """
    Process a single PDF file and add it to ChromaDB.
    
    Args:
        pdf_path (str): Path to the PDF file.
    """
    if not os.path.exists(pdf_path):
        print(f"⚠️ File {pdf_path} not found!")
        return

    doc_id = os.path.basename(pdf_path).replace(".pdf", "")  
    add_pdf_to_collection(pdf_path, doc_id)

# Example usage: Process all PDFs in 'data/pdf' folder
pdf_path = os.path.join(os.getcwd(), "data/pdf")
for filename in os.listdir(pdf_path):
    file_path = os.path.join(pdf_path, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
        process_pdf_file(file_path)

def query_chromadb(query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    print("🔖 Metadata:", results.get("metadatas", []))  
    print("📌 Retrieved IDs:", results.get("ids", [])) 
    return results.get("documents", []), results.get("metadatas", [])

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
    # print("######## RAG PIPELINE ########")
    # print(retrieved_docs)
    # print(metadata)
    context = " ".join([" ".join(doc) for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

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
    - DO NOT mention where the information was retrieved from or reference specific sections of the document.

    Answer:
    """
    
    # print("######## Augmented Prompt ########")
    # print(augmented_prompt)

    response = query_ollama(augmented_prompt)
    return response

# # Example usage
# # Define a query to test the RAG pipeline
# query = "What are the factors that make us vulnerable to cyber attacks?" 
# response = rag_pipeline(query)
# print("######## Response from LLM ########\n", response)

def chatbot():
    print("🟢 RAG Chatbot is running. Type 'exit' to quit.")
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() == "exit":
            print("🛑 Chatbot session ended.")
            break
        response = rag_pipeline(user_input)
        print("🤖 Chatbot:", response, "\n")

if __name__ == "__main__":
    chatbot()
