from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
import re
import time
import json
import os

HISTORY_FILE = "MILESTONE_history.json"
def save_history_json(user_id, query, answer):
    # Load existing data or initialize empty
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    entry = {
        "analyst_id": user_id,
        "query": query,
        "answer": answer,
        "timestamp": int(time.time())
    }

    if user_id not in data:
        data[user_id] = []

    data[user_id].append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# LLM
llm = ChatOllama(model="mistral", temperature=0.3)
print("\n=== STEP 1: LOAD DATA ===")
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded documents: {len(docs)}")
print("Sample:", docs[0].page_content[:200], "\n")

print("\n=== STEP 2: CHUNKING ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
splits = splitter.split_documents(docs)
print(f"Total chunks: {len(splits)}\n")

print("\n=== STEP 3: FAISS INDEX ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("FAISS ready.\n")

print("\n=== STEP 4: MANUAL HYBRID RETRIEVAL ===")
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4
def hybrid_search(query):
    """Combine FAISS + BM25 results manually."""
    vec = vector_retriever._get_relevant_documents(query, run_manager=None)
    bm = bm25._get_relevant_documents(query, run_manager=None)
    combined = vec + bm
    unique = {doc.page_content: doc for doc in combined}.values()
    return list(unique)[:4]
print("Hybrid retriever ready.\n")

print("\n=== STEP 5: RAG CHAIN ===")
prompt = ChatPromptTemplate.from_template("""
You are a SOC Analyst Assistant.
Return the response STRICTLY IN JSON:
{{
  "root_cause": "",
  "similar_incidents": "",
  "analysis": "",
  "recommended_steps": "",
  "entities": {entities},
  "Summary": ""                                       
}}

### Retrieved Context:
{context}

### Analyst Memory:
{history}

### Entities:
{entities}

### Query:
{question}

Respond only in JSON. No extra text.
""")

prompt1 = ChatPromptTemplate.from_template("""
You are a SOC Analyst Assistant.
Return the response STRICTLY IN JSON:
{{
  "Summary": ""                                       
}}

Respond only in JSON. No extra text.
""")

parser = StrOutputParser()
chain = (
    {
        "context": lambda q: "\n".join([d.page_content for d in hybrid_search(q["question"])]),
        "question": lambda q: q["question"],
        "entities": lambda q: q["entities"],
        "history": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)
chain1 = (
    {
        "context": lambda q: "\n".join([d.page_content for d in hybrid_search(q["question"])]),
        "question": lambda q: q["question"],
        "entities": lambda q: q["entities"],
        "history": RunnablePassthrough(),
    }
    | prompt1
    | llm
    | parser
)
print("RAG chain ready.\n")

print("\n=== STEP 6: MEMORY ===")
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
memory_chain1 = RunnableWithMessageHistory(
    chain1,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
print("Memory wrapper ready.\n")

def extract_entities(query, context=""):
    text = query + " " + context
    return {
        "ips": re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text),
        "mitre": re.findall(r"T\d{4}", text),
        "os": re.findall(r"(Windows|Linux|Ubuntu|CentOS|macOS)", text),
        "severity": re.findall(r"(Low|Medium|High|Critical)", text)
    }

print("\n=== STEP 7: CONSOLE LOOP ===\n")
while True:
    raw = input("Enter: <analyst_id> <query> (or q): ")
    if raw == "q":
        break
    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Format: analyst42 suspicious ssh activity")
        continue
    analyst_id, query = parts
    context_docs = hybrid_search(query)
    context_text = "\n".join([d.page_content for d in context_docs])
    entities = extract_entities(query, context_text)
    response = memory_chain.invoke(
        {
            "question": query,
            "entities": entities,           # ← USE REAL DICT
        },
        config={"session_id": analyst_id}
    )
    response1 = memory_chain1.invoke(
        {
            "question": query,
            "entities": entities,           # ← USE REAL DICT
        },
        config={"session_id": analyst_id}
    )
    print("\n=== SOC JSON RESPONSE ===")
    print(response)
    save_history_json(analyst_id, query, response1)
    print("\n---------------------------------------------\n")