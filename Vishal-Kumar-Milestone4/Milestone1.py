# ================================================================
# SOC RAG Assistant - Assessment Template (Students: Fill TODOs)
# ================================================================

# ===========
# STEP 0: IMPORTS (DO NOT REMOVE — Add only if needed)
# ===========
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
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import re
import json

# LLM Setup
llm = ChatOllama(model="mistral", temperature=0.3)


# STEP 1: LOAD DATA
print("\n=== STEP 1: LOAD DATA ===")
# TODO: Load security_incidents.txt using TextLoader
# TODO: Print doc count + sample record
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded {len(docs)} documents. Sample: {docs[0].page_content[:200]}...")


# ===========
# STEP 2: CHUNK DATA
# ===========
print("\n=== STEP 2: CHUNKING ===")
# TODO: Implement RecursiveCharacterTextSplitter
# TODO: Produce splits
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks.")


# ===========
# STEP 3: EMBEDDINGS + FAISS
# ===========
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
# TODO: Create embeddings using MiniLM-L6-v2
# TODO: Create FAISS vectorstore
# TODO: Create vector retriever (k=4)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ===========
# BONUS: HYBRID RETRIEVAL
# ===========
print("\n=== BONUS: HYBRID RETRIEVAL ===")
# TODO: Create BM25 retriever
# TODO: Create EnsembleRetriever with weights [0.7, 0.3]
bm25_retriever = BM25Retriever.from_documents(splits)
hybrid_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.7, 0.3])
print("Hybrid retriever ready.\n")


# ===========
# BONUS: ENTITY EXTRACTION
# ===========
def extract_entities(query: str, context_docs):
    entities = {}
    text = query + " " + " ".join([doc.page_content for doc in context_docs])
    # IP addresses
    ips = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)
    if ips:
        entities['IPs'] = list(set(ips))
    # OS
    os_patterns = ['Windows', 'Linux', 'macOS', 'Android', 'iOS', 'Chrome OS']
    for os in os_patterns:
        if os.lower() in text.lower():
            entities['OS'] = os
            break
    # Hostname
    hostnames = re.findall(r'Hostname: (\w+)', text)
    if hostnames:
        entities['Hostname'] = hostnames[0]
    # MITRE
    mitre = re.findall(r'T\d{4}', text)
    if mitre:
        entities['MITRE'] = list(set(mitre))
    # Severity
    severity = re.findall(r'Severity: (\w+)', text)
    if severity:
        entities['Severity'] = severity[0]
    return entities


# ===========
# BONUS: THREAT SCORE
# ===========
def calculate_threat_score(incident):
    score = 0
    if 'T1486' in incident:  # Ransomware
        score += 10
    if 'Critical' in incident:
        score += 5
    elif 'High' in incident:
        score += 3
    elif 'Medium' in incident:
        score += 2
    if 'brute-force' in incident.lower():
        score += 2
    if 'PowerShell' in incident.lower():
        score += 2
    if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', incident):
        score += 1
    return score


# ===========
# STEP 4: PROMPT + LCEL RAG CHAIN
# ===========
print("\n=== STEP 4: RAG CHAIN ===")

# TODO: Build prompt template (you must include placeholders for context, question, entities, history)
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

History: {history}

Entities: {entities}

Question: {question}
""")

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chain_func(input_dict):
    context_docs = hybrid_retriever.invoke(input_dict["question"])
    entities = extract_entities(input_dict["question"], context_docs)
    entities_str = ", ".join([f"{k}: {v}" for k, v in entities.items()]) if entities else "None"
    threat_scores = [calculate_threat_score(doc.page_content) for doc in context_docs]
    scores_str = ", ".join([f"Doc{i+1}: {score}" for i, score in enumerate(threat_scores)])
    prompt_input = {
        "context": format_docs(context_docs),
        "question": input_dict["question"],
        "history": "\n".join([f"{msg.type}: {msg.content}" for msg in input_dict.get("history", [])]),
        "entities": entities_str
    }
    response = (prompt | llm | parser).invoke(prompt_input)
    # BONUS: Structured Output
    structured = {
        "retrieved_context": [doc.page_content for doc in context_docs],
        "injected_user_memory": input_dict.get("history", []),
        "injected_entity_memory": entities,
        "final_llm_answer": response,
        "threat_scores": threat_scores
    }
    return json.dumps(structured, indent=2)

from langchain_core.runnables import RunnableLambda
chain = RunnableLambda(chain_func)
print("RAG chain constructed.\n")


# ===========
# BONUS: TOOL
# ===========
print("\n=== BONUS: TOOL ===")

@tool
def threat_enrich(ip: str):
    """Students: Return mock threat intel for IP"""
    # TODO: Implement enrichment logic
    return f"Threat info for {ip}: Malicious activity detected, block immediately."

# TODO: Bind tool to LLM
llm_with_tools = llm.bind_tools([threat_enrich])


# ===========
# STEP 5: MEMORY (RunnableWithMessageHistory)
# ===========
print("\n=== STEP 5: MEMORY ===")

store = {}  # DO NOT CHANGE

def get_session_history(session_id: str):
    # TODO: Create InMemoryChatMessageHistory if missing
    # TODO: Return the history object
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# TODO: Wrap chain with RunnableWithMessageHistory
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
print("Memory wrapper ready.\n")


# ===========
# STEP 6: ENTITY EXTRACTION (BONUS)
# ===========
# Already defined above


# ===========
# STEP 7: INTERACTIVE CONSOLE LOOP
# ===========
print("\n=== STEP 7: CONSOLE LOOP ===")

while True:
    raw = input("Enter: <analyst_id> <query> (or q): ")
    if raw == "q":
        break
    
    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Format: analyst42 suspicious ssh activity")
        continue

    analyst_id, query = parts

    # TODO: Retrieve entities from query or context
    # entities = {}  # Placeholder, but we extract in chain_func

    # TODO: Invoke the memory-wrapped RAG chain
    config = {"configurable": {"session_id": analyst_id}}
    response = memory_chain.invoke({"question": query}, config=config)

    # TODO: Print the response
    print(f"Response for {analyst_id}: {response}")
