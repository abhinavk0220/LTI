from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory

import re
import json

# 1. Load CSV
loader = CSVLoader("incidents.csv")
docs = loader.load()

# 2. Split Documents
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# 3. Embeddings (MiniLM)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Vector Store (FAISS)
vector_store = FAISS.from_documents(chunks, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 5. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# 6. Hybrid Retriever
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# 7. LLM (MISTRAL)
llm = ChatOllama(model="mistral", temperature=0.7)

# 8. Tool for internal knowledge base lookup
@tool
def kb_lookup(query: str) -> str:
    """Lookup internal knowledge base for common security incident resolutions."""
    fixes = {
        "outlook crash": "Restart Outlook → Repair Office → Delete corrupt OST.",
        "ransomware": "Isolate host → Trigger IR plan → Restore from backup.",
        "ssh brute-force": "Block IP → Enable fail2ban → Reset credentials.",
    }
    for key in fixes:
        if key in query.lower():
            return fixes[key]
    return "No KB entry found."

llm_tools = llm.bind_tools([kb_lookup])

# 9. Entity Extraction
def extract_entities(text):
    entities = {}
    ip_match = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    mitre_match = re.findall(r"T\d{4}(?:\.\d+)?", text)
    severity_match = re.findall(r"\b(Critical|High|Medium|Low)\b", text)
    os_match = re.findall(r"\bWindows|Linux|Ubuntu|CentOS|Debian|RedHat|macOS\b", text)
    host_match = re.findall(r"Host(?:name)?=(\S+)", text)
    if ip_match: entities["IP"] = ip_match
    if mitre_match: entities["MITRE"] = mitre_match
    if severity_match: entities["Severity"] = severity_match
    if os_match: entities["OS"] = os_match
    if host_match: entities["Hostname"] = host_match
    return entities

# 10. Threat Score
def compute_threat_score(entities, query):
    score = 0
    if "Critical" in entities.get("Severity", []): score += 30
    if any("T1059" in t or "PowerShell" in query for t in entities.get("MITRE", [])): score += 20
    if any("brute" in query.lower() or "SSH" in query for t in entities.get("MITRE", [])): score += 20
    if entities.get("IP"): score += 10
    return score

# 11. Threat Category Classification
def classify_threat(query):
    if "PowerShell" in query or "script" in query:
        return "Execution"
    elif "login" in query or "credential" in query:
        return "Credential Access"
    elif "ransomware" in query or "encryption" in query:
        return "Impact"
    elif "data transfer" in query or "exfil" in query:
        return "Exfiltration"
    return "Unknown"

# 12. Analyst Preference Memory
user_preferences = {}

def update_preferences(user_id, entities):
    prefs = user_preferences.get(user_id, {})
    if "OS" in entities:
        prefs["preferred_os"] = entities["OS"][0]
    if "Severity" in entities:
        prefs["severity_focus"] = entities["Severity"][0]
    user_preferences[user_id] = prefs

def get_preferences(user_id):
    return user_preferences.get(user_id, {})

# 13. Prompt Template (Structured JSON Output)
prompt_template = """
You are a SOC analyst assistant.

Respond in JSON format with keys:
- resolution: short fix or recommendation
- similar_cases: list of matching incidents
- threat_summary: key entities and threat score

Context:
{context}

Chat History:
{chat_history}

Entities: {entities}
Threat Score: {score}
Threat Category: {category}
Analyst Preferences: {preferences}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question", "entities", "score", "category", "preferences"],
    template=prompt_template
)

# 14. Conversational Retrieval Chain
base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_tools,
    retriever=hybrid_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# 15. User memory store
user_memory_store = {}

def get_user_memory(user_id: str):
    if user_id not in user_memory_store:
        user_memory_store[user_id] = InMemoryChatMessageHistory()
    return user_memory_store[user_id]

# 16. Process query
def process_query(user_id: str, query: str):
    user_memory = get_user_memory(user_id)
    entities = extract_entities(query)
    score = compute_threat_score(entities, query)
    category = classify_threat(query)
    update_preferences(user_id, entities)
    preferences = get_preferences(user_id)

    response = base_rag_chain.invoke(
        {
            "question": query,
            "chat_history": user_memory.messages,
            "entities": str(entities),
            "score": score,
            "category": category,
            "preferences": str(preferences)
        }
    )

    user_memory.add_user_message(query)
    user_memory.add_ai_message(response["answer"])

    try:
        return json.loads(response["answer"])
    except:
        return {"raw_response": response["answer"]}