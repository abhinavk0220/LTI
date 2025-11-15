from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory,RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import re

llm = ChatOllama(model="mistral",temperature=0.3)

loader = TextLoader("security_incidents.txt")
docs = loader.load()

splitting = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitting.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

texts_only = [d.page_content for d in splits]
vectorstore = FAISS.from_texts(texts_only,embedding)

faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4

hybrid = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25],
    weights=[0.7, 0.3]
)


prompt = ChatPromptTemplate.from_template(
    """
You are a SOC Analyst Assistant – Incident RAG Investigation System. Use all information carefully.

Retrieved Context:
{context}

Extracted Entities:
{entities}

Analyst History:
{history}

Question:
{question}

Answer in a clear, SOC-analysis style.
- Recommend resolutions
- Retrieve similar past incidents
- Recall analyst-specific preferences across sessions
- Extract important entities (IPs, OS, MITRE tags)
- Provide enriched threat analysis
All these in short
""")

parser = StrOutputParser()

def extract_entities(query: str, context: str):
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", query + context)
    mitre = re.findall(r"T\d{4}", query + context)
    sev = re.findall(r"high|medium|low|critical", query.lower())

    return {
        "ips": ips,
        "mitre": mitre,
        "severity": sev
    }

def build_context(x, config):
    return hybrid.invoke(x["question"], config)


def build_entities(x, config):
    ctx = hybrid.invoke(x["question"], config)
    joined = " ".join([c.page_content for c in ctx])
    return extract_entities(x["question"], joined)


chain = (
    {
        "context": build_context,
        "entities": build_entities,
        "history": RunnablePassthrough(),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)

@tool
def threat_enrich(ip: str):
    """Mock threat intelligence lookup"""
    return f"Threat intel for {ip}: Suspicious reputation score."

llm_with_tools = llm.bind_tools([threat_enrich])


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)




while True:
    raw = input("Enter: <analyst_id> <query> (or q): ")

    if raw == "q":
        break

    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Format: analyst42 suspicious ssh activity")
        continue

    analyst_id, query = parts

    config = {"configurable": {"session_id": analyst_id}}

    response = memory_chain.invoke({"question": query}, config)

    print("\n=== RESPONSE ===")
    print(response)
    print("================\n")

