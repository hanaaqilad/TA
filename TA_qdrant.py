from langchain_ollama import OllamaEmbeddings, OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import shutil
import pypdf  
import re
import json

# Define the LLM model to be used
llm_model = "llama3.1:8b"

# Initialize Qdrant
qdrant_client = QdrantClient(":memory:")  # Change to "localhost" or Qdrant Cloud URL if needed

# Define embedding function using Ollama
embedding = OllamaEmbeddings(
    model=llm_model,
    base_url="http://localhost:11434"
)

if not qdrant_client.collection_exists("general_knowledge_base"):
    qdrant_client.create_collection(
        collection_name="general_knowledge_base",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

if not qdrant_client.collection_exists("qa_knowledge_base"):
    qdrant_client.create_collection(
        collection_name="qa_knowledge_base",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    reader = pypdf.PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text.strip()

def extract_qa_from_pdf(pdf_path):
    """Extracts structured QA pairs from a PDF."""
    knowledge_base = []
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            raw_text = page.extract_text()
            if raw_text:
                text = " ".join(raw_text.split("\n"))
                pattern = r"Question:\s*(.*?)\?\s*Risk:\s*\{'yes':\s*'(.*?)',\s*'no':\s*'(.*?)'\}"
                matches = re.findall(pattern, text)
                for match in matches:
                    question, yes_risk, no_risk = match
                    knowledge_base.append({
                        "question": question.strip() + "?",
                        "risk": {"yes": yes_risk.strip(), "no": no_risk.strip()}
                    })
    return knowledge_base

def add_pdf_to_qdrant(pdf_path, doc_id):
    """Processes PDF and adds text embeddings to Qdrant."""
    if "qa_knowledge_base" in doc_id:
        knowledge_base = extract_qa_from_pdf(pdf_path)
        if not knowledge_base:
            print(f"⚠️ No QA extracted from {pdf_path}. Skipping.")
            return
        for i, item in enumerate(knowledge_base):
            vector = embedding.embed_documents([item["question"]])[0]
            qdrant_client.upsert(
                collection_name="qa_knowledge_base",
                points=[
                    PointStruct(id=i, vector=vector, payload={"risk": json.dumps(item["risk"])})
                ]
            )
        print(f"✅ QA from '{doc_id}' added to Qdrant.")
    else:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"⚠️ No text extracted from {pdf_path}. Skipping.")
            return
        vector = embedding.embed_documents([text])[0]
        qdrant_client.upsert(
            collection_name="general_knowledge_base",
            points=[
                PointStruct(id=doc_id, vector=vector, payload={"content": text})
            ]
        )
        print(f"✅ PDF '{doc_id}' added to general knowledge base in Qdrant.")

def process_pdf_file(pdf_path):
    """Processes a PDF file and adds it to Qdrant."""
    if not os.path.exists(pdf_path):
        print(f"⚠️ File {pdf_path} not found!")
        return
    doc_id = os.path.basename(pdf_path).replace(".pdf", "")  
    add_pdf_to_qdrant(pdf_path, doc_id)

# Process PDFs in "data/pdf" folder
pdf_folder = os.path.join(os.getcwd(), "data/pdf")
for filename in os.listdir(pdf_folder):
    file_path = os.path.join(pdf_folder, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
        process_pdf_file(file_path)

def query_qdrant(collection_name, query_text, n_results=3):
    """Retrieves the most relevant documents from Qdrant."""
    query_vector = embedding.embed_documents([query_text])[0]
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=n_results
    )
    
    print("📌 Retrieved IDs:", [res.id for res in results]) 
    return [res.payload.get("content", "") for res in results], [res.payload for res in results]

def query_ollama(prompt):
    """Queries the Ollama LLM for responses."""
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

def rag_pipeline(collection_name, query_text):
    """Performs Retrieval-Augmented Generation (RAG) using Qdrant and Ollama."""
    retrieved_docs, metadata = query_qdrant(collection_name, query_text)
    context = " ".join(retrieved_docs) if retrieved_docs else "No relevant documents found."

    augmented_prompt = f"""
    You are an AI assistant that provides precise and relevant answers based on the retrieved context.

    Context:
    {context}

    Question:
    {query_text}

    Guidelines:
    - Keep responses concise yet informative.
    - If no relevant data is found, clearly state it.
    - Provide additional insights only when necessary.

    Answer:
    """
    
    response = query_ollama(augmented_prompt)
    return response

def chatbot():
    """Interactive chatbot using Qdrant for retrieval and Ollama for response generation."""
    print("🤖 Welcome! Choose an option:")
    print("1. Consult about cybersecurity risks")
    print("2. Risk assessment quiz")

    choice = input("Choose (1 or 2): ")

    if choice == "1":
        print("🟢 Ask me anything about cybersecurity risks! Type 'exit' to quit.")
        while True:
            user_input = input("👤 You: ")
            if user_input.lower() == "exit":
                print("🛑 Chatbot session ended.")
                break
            response = rag_pipeline("general_knowledge_base", user_input)
            print("🤖 Chatbot:", response, "\n")

    elif choice == "2":
        print("📖 Loading risk assessment knowledge base...")
        all_docs = qdrant_client.scroll("qa_knowledge_base", limit=50)
        questions = [doc.payload.get("content", "") for doc in all_docs[0]]

        if not questions:
            print("⚠️ No questions found in the risk assessment knowledge base.")
            return

        user_answers = []
        print("📝 Answer the following Yes/No questions for risk analysis.")

        for question in questions:
            answer = input(f"🤖 {question} (yes/no): ").strip().lower()
            while answer not in ["yes", "no"]:
                print("❌ Please answer with 'yes' or 'no'.")
                answer = input(f"🤖 {question} (yes/no): ").strip().lower()
            user_answers.append(answer)

        print("\n📊 Your cybersecurity risk assessment is complete.\n")

    else:
        print("❌ Invalid choice. Please restart and enter 1 or 2.")

if __name__ == "__main__":
    chatbot()
