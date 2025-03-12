from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import shutil
import pypdf  
import re
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Define the LLM model to be used
llm_model = "llama3.1:8b"

# Configure ChromaDB and reset every run
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
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

# # Define a collection for the RAG workflow
# collection_name = "rag_collection_demo"
# collection = chroma_client.get_or_create_collection(
#     name=collection_name,
#     metadata={"description": "A collection for RAG with Ollama - Demo"},
#     embedding_function=embedding  
# )

# Define collections for different knowledge bases
general_collection = chroma_client.get_or_create_collection(
    name="general_knowledge_base",
    metadata={"description": "General cybersecurity knowledge base"},
    embedding_function=embedding  
)

risk_assessment_collection = chroma_client.get_or_create_collection(
    name="qa_knowledge_base",
    metadata={"description": "Risk assessment knowledge base"},
    embedding_function=embedding  
)

def extract_text_from_pdf(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        text += extracted_text + "\n" if extracted_text else ""  
    return text.strip()

def extract_text_from_images(pdf_path):
    """
    Extracts text from an image-based PDF using OCR.
    """
    images = convert_from_path(pdf_path)
    extracted_text = ""
    print("YEEEEEEEEEEEEE 💀💀💀💀💀💀💀💀💀💀💀💀💀")
    for img in images:
        text = pytesseract.image_to_string(img, lang="eng")
        extracted_text += text + "\n"

    return extracted_text.strip()

def extract_qa_from_pdf(pdf_path):
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

def add_pdf_to_collection(pdf_path, doc_id):
    if "qa_knowledge_base" in doc_id:
        knowledge_base = extract_qa_from_pdf(pdf_path)
        if not knowledge_base:
            print(f"⚠️ No QA extracted from {pdf_path}. Skipping.")
            return
        for i, item in enumerate(knowledge_base):
            risk_assessment_collection.add(documents=[item["question"]], 
                                           ids=[f"{doc_id}_{i}"], 
                                           metadatas=[{"risk": json.dumps(item["risk"])}])
            # print(f"✅ QA '{item['question']}' added to risk assessment collection.")

        all_docs = risk_assessment_collection.get()
        print("📌 Total QA in QA collection:", len(all_docs["ids"]))
        print("📝 Stored QA IDs:", all_docs["ids"])

    else:
        text = extract_text_from_pdf(pdf_path)

        if not text:  # If no text found, try OCR
            print(f"🔍 No text detected in {doc_id}, using OCR...")
            text = extract_text_from_images(pdf_path)

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

        general_collection.add(documents=[text], ids=[doc_id])
        print(f"✅ PDF '{doc_id}' added to general collection.")

        all_docs = general_collection.get()
        print("📌 Total documents in general collection:", len(all_docs["ids"]))
        print("📝 Stored document IDs:", all_docs["ids"])


def process_pdf_file(pdf_path):
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

def assess_risk(user_answers):
    risk_score = 0
    risk_weights = {"low": 0, "moderate": 1, "high": 2}
    
    all_docs = risk_assessment_collection.get()
    metadatas = all_docs.get("metadatas", [])
    
    for i, ans in enumerate(user_answers):
        if i >= len(metadatas):
            continue  # Skip if metadata index is out of range
        
        try:
            risk_data = json.loads(metadatas[i].get("risk", "{}"))
            risk_category = risk_data.get(ans.lower(), "moderate")
        except json.JSONDecodeError:
            risk_category = "moderate"
        
        risk_score += risk_weights.get(risk_category.lower(), 1)
    
    if risk_score >= len(user_answers) * 1.5:
        return "HIGH"
    elif risk_score >= len(user_answers) * 0.5:
        return "MODERATE"
    else:
        return "LOW"
    
    # risk_score = 0
    # risk_weights = {"low": 0, "moderate": 1, "high": 2}

    # for i, ans in enumerate(user_answers):
    #     doc = risk_assessment_collection.get(ids=[str(i)])
        
    #     if "metadatas" in doc and doc["metadatas"]:
    #         try:
    #             risk_data = json.loads(doc["metadatas"][0]["risk"])
    #             risk_category = risk_data.get(ans.lower(), "moderate")
    #         except json.JSONDecodeError:
    #             risk_category = "moderate"
    #     else:
    #         risk_category = "moderate"

    #     risk_score += risk_weights.get(risk_category.lower(), 1)  # Default moderate jika tidak ditemukan

    # # Tentukan risk level berdasarkan skor total
    # if risk_score >= len(user_answers) * 1.5:  # Lebih dari 75% pertanyaan berisiko tinggi
    #     return "HIGH"
    # elif risk_score >= len(user_answers) * 0.5:  # Lebih dari 25% pertanyaan berisiko sedang atau tinggi
    #     return "MODERATE"
    # else:
    #     return "LOW"
    


def query_chromadb(collection, query_text, n_results=3):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    print("🔖 Metadata:", results.get("metadatas", []))  
    print("📌 Retrieved IDs:", results.get("ids", [])) 
    return results.get("documents", []), results.get("metadatas", [])

def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model)
    return llm.invoke(prompt)

# RAG pipeline: Combine ChromaDB and Ollama for RAG
def rag_pipeline(collection, query_text):
    """
    Perform Retrieval-Augmented Generation (RAG) by combining ChromaDB and Ollama.
    
    Args:
        query_text (str): The input query.
    
    Returns:
        str: The generated response from Ollama augmented with retrieved context.
    """
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(collection, query_text)
    context = " ".join([" ".join(doc) for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

    # print("######## RAG PIPELINE ########")
    # print(retrieved_docs)
    # print(metadata)

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
    - Mention where the information was retrieved from or reference specific sections of the document.

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

# def chatbot():
#     print("🟢 RAG Chatbot is running. Type 'exit' to quit.")
#     while True:
#         user_input = input("👤 You: ")
#         if user_input.lower() == "exit":
#             print("🛑 Chatbot session ended.")
#             break
#         response = rag_pipeline(user_input)
#         print("🤖 Chatbot:", response, "\n")

def chatbot():
    print("🤖 Welcome! Choose an option:")
    print("1. Consult about cybersecurity risks --- You can ask anything related to cyber risks")
    print("2. Risk assessment --- Assess your risk with multiple questions and get score!")
    
    choice = input("Which one you choose (1 or 2)? ")
    
    if choice == "1":
        print("🟢 Ask me anything about cybersecurity risks! Type 'exit' to quit.")
        while True:
            user_input = input("👤 You: ")
            if user_input.lower() == "exit":
                print("🛑 Chatbot session ended.")
                break
            response = rag_pipeline(general_collection, user_input)
            print("🤖 Chatbot:", response, "\n")
    elif choice == "2":
        print("📖 Loading risk assessment knowledge base...")
        all_docs = risk_assessment_collection.get()
        questions = all_docs.get("documents", [])
        
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
        
        risk_level = assess_risk(user_answers)
        print(f"\n📊 Your cybersecurity risk level is: {risk_level.upper()}\n")

        # print("📖 Loading risk assessment knowledge base...")
        # user_answers = []
        # print("📝 Answer the following Yes/No questions for risk analysis.")
        # for i in range(len(risk_assessment_collection.get()["ids"])):
        #     print(risk_assessment_collection.get())
        #     print(risk_assessment_collection.get()["ids"])
        #     question = risk_assessment_collection.get(ids=[str(i)])[0]
        #     answer = input(f"🤖 {question} (yes/no): ").strip().lower()
        #     while answer not in ["yes", "no"]:
        #         print("❌ Please answer with 'yes' or 'no'.")
        #         answer = input(f"🤖 {question} (yes/no): ").strip().lower()
        #     user_answers.append(answer)
        
        # risk_level = assess_risk(user_answers)
        # print(f"\n📊 Your cybersecurity risk level is: {risk_level.upper()}\n")
    else:
        print("❌ Invalid choice. Please restart and enter 1 or 2.")

if __name__ == "__main__":
    chatbot()
