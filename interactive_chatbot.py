import os
import shutil
import chromadb
import pypdf
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Model & database setup
llm_model = "llama3.1:8b"
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
if os.path.exists(chroma_db_path):
    shutil.rmtree(chroma_db_path)
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Custom embedding function
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings
    
    def __call__(self, input):
        input = [input] if isinstance(input, str) else input
        embeddings = self.langchain_embeddings.embed_documents(input)
        return [embeddings] if isinstance(embeddings[0], float) else embeddings

embedding = ChromaDBEmbeddingFunction(OllamaEmbeddings(model=llm_model, base_url="http://localhost:11434"))

# Knowledge base setup
collection_name = "cyber_risk_assessment"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "Cybersecurity risk assessment knowledge base"},
    embedding_function=embedding
)

import re

def extract_knowledge_from_pdf(pdf_path):
    knowledge_base = []
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            print(f"📄 Page {page_num + 1} extracted text (before processing):\n{raw_text}\n{'-'*40}")

            if raw_text:
                # Gabungkan semua teks menjadi satu baris
                text = " ".join(raw_text.split("\n"))
                print(f"📝 Processed text:\n{text}\n{'-'*40}")  # Debug

                # Gunakan regex untuk menangkap pertanyaan dan risiko
                pattern = r"Question:\s*(.*?)\?\s*Risk:\s*\{'yes':\s*'(.*?)',\s*'no':\s*'(.*?)'\}"
                matches = re.findall(pattern, text)

                for match in matches:
                    question, yes_risk, no_risk = match
                    knowledge_base.append({
                        "question": question.strip() + "?",
                        "risk": {"yes": yes_risk.strip(), "no": no_risk.strip()}
                    })

    return knowledge_base

pdf_path = os.path.join(os.getcwd(), "data/pdf/qa_knowledge_base.pdf")
print(f"📂 Checking PDF path: {pdf_path}")

knowledge_base = extract_knowledge_from_pdf(pdf_path)
if not knowledge_base:
    print("⚠️ Error: No questions extracted from the PDF. Please check the file format.")
    exit()

import json
# Adding knowledge base to ChromaDB
for i, item in enumerate(knowledge_base):
    collection.add(documents=[item["question"]], 
               ids=[str(i)], 
               metadatas=[{"risk": json.dumps(item["risk"])}])  # Convert dict to JSON string

import json  # Untuk parsing JSON string ke dictionary

def assess_risk(user_answers):
    risk_score = 0
    risk_weights = {"low": 0, "moderate": 1, "high": 2}

    for i, ans in enumerate(user_answers):
        doc = collection.get(ids=[str(i)])
        
        if "metadatas" in doc and doc["metadatas"]:
            try:
                risk_data = json.loads(doc["metadatas"][0]["risk"])
                risk_category = risk_data.get(ans.lower(), "moderate")
            except json.JSONDecodeError:
                risk_category = "moderate"
        else:
            risk_category = "moderate"

        risk_score += risk_weights.get(risk_category.lower(), 1)  # Default moderate jika tidak ditemukan

    # Tentukan risk level berdasarkan skor total
    if risk_score >= len(user_answers) * 1.5:  # Lebih dari 75% pertanyaan berisiko tinggi
        return "HIGH"
    elif risk_score >= len(user_answers) * 0.5:  # Lebih dari 25% pertanyaan berisiko sedang atau tinggi
        return "MODERATE"
    else:
        return "LOW"

def chatbot():
    print("\n🔐 Cybersecurity Risk Assessment Chatbot 🔐\n")
    user_answers = []
    
    for item in knowledge_base:
        answer = input(f"{item['question']} (yes/no): ").strip().lower()
        while answer not in ["yes", "no"]:
            print("❌ Please answer with 'yes' or 'no'.")
            answer = input(f"{item['question']} (yes/no): ").strip().lower()
        user_answers.append(answer)
    
    risk_level = assess_risk(user_answers)
    print(f"\n📊 Your cybersecurity risk level is: {risk_level.upper()}\n")

if __name__ == "__main__":
    chatbot()