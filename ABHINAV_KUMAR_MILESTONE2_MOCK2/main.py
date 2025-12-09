import os
import sys
import json
import datetime
from typing import List, Dict, TypedDict, Optional, Any
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# --- Configuration ---
# No API Key required for local execution.
# Using HuggingFaceEmbeddings (all-MiniLM-L6-v2)

POLICY_FILE_PATH = "data/policies/coverage_rules.md"

# --- 1. Data Loading & RAG Setup ---

def setup_rag_system():
    """Loads policy documents and creates a vector store."""
    if not os.path.exists(POLICY_FILE_PATH):
        raise FileNotFoundError(f"Policy file not found at {POLICY_FILE_PATH}")

    loader = TextLoader(POLICY_FILE_PATH)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    print("Initializing Local Embeddings (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Global VectorStore (initialized later)
vectorstore = None

def get_policy_clauses(query: str) -> List[str]:
    """Retrieves relevant policy clauses based on the query."""
    if vectorstore is None:
        return ["(Error) Vectorstore not initialized."]
    
    docs = vectorstore.similarity_search(query, k=3)
    return [d.page_content for d in docs]

# --- 2. State Definition ---

class AgentState(TypedDict):
    claim_text: str
    parsed_fields: Dict[str, Any]
    policy_clauses: List[str]
    damage_cost: float
    missing_docs: List[str]
    fraud_score: int
    approval_status: str
    settlement_estimate: float
    final_report: str
    iteration_count: int # To prevent infinite loops

# --- 3. Nodes ---

def parse_claim_details(state: AgentState):
    """Extracts structured data from claim text."""
    print("\n--- Node: Parse Claim Details ---")
    claim_text = state["claim_text"]
    
    # Mocking LLM extraction for reliability without API key, 
    # or if we want to ensure specific output for the demo.
    # In a full production system, use ChatOpenAI here.
    
    # Simple rule-based extraction for demo purposes if LLM fails or simple text
    parsed = {}
    lower_text = claim_text.lower()
    
    # Extract Cost
    import re
    cost_match = re.search(r"rs\.?\s?(\d+([\d,]+)?)", lower_text)
    if cost_match:
        parsed["repair_cost"] = float(cost_match.group(1).replace(",", ""))
    else:
        parsed["repair_cost"] = 50000.0 # Default/Fallback

    # Extract Incident Type
    if "accident" in lower_text:
        parsed["incident_type"] = "Accident"
    elif "theft" in lower_text:
        parsed["incident_type"] = "Theft"
    else:
        parsed["incident_type"] = "Unknown"

    # Extract Docs
    docs = []
    if "license" in lower_text: docs.append("driving_license")
    if "rc" in lower_text or "registration" in lower_text: docs.append("rc_copy")
    if "fir" in lower_text: docs.append("fir_copy")
    if "policy" in lower_text: docs.append("policy_doc")
    parsed["submitted_docs"] = docs
    
    parsed["policy_id"] = "POL123456" # Mock

    return {"parsed_fields": parsed, "damage_cost": parsed.get("repair_cost", 0.0)}

def retrieve_rules_rag(state: AgentState):
    """Retrieves relevant policy rules."""
    print("\n--- Node: Retrieve Rules RAG ---")
    incident_type = state["parsed_fields"].get("incident_type", "")
    clauses = get_policy_clauses(f"coverage for {incident_type} and deductibles")
    return {"policy_clauses": clauses}

def check_document_completeness(state: AgentState):
    """Checks for missing documents."""
    print("\n--- Node: Check Document Completeness ---")
    required_docs = ["driving_license", "policy_doc"]
    incident_type = state["parsed_fields"].get("incident_type", "")
    
    if incident_type == "Theft":
        required_docs.append("fir_copy")
    
    submitted = state["parsed_fields"].get("submitted_docs", [])
    missing = [doc for doc in required_docs if doc not in submitted]
    
    return {"missing_docs": missing}

def fraud_detection_agent(state: AgentState):
    """Calculates fraud score based on heuristics."""
    print("\n--- Node: Fraud Detection ---")
    score = 0
    claim_text = state["claim_text"].lower()
    
    # Rule 1: Late reporting (Mock check)
    if "days later" in claim_text:
        days = int(re.search(r"(\d+) days later", claim_text).group(1))
        if days > 7:
            score += 30
            
    # Rule 2: High amount
    if state["damage_cost"] > 100000:
        score += 20
        
    # Rule 3: No police report for theft
    if state["parsed_fields"].get("incident_type") == "Theft" and "fir_copy" in state["missing_docs"]:
        score += 50

    # Rule 4: Weekend incident (Mock)
    if "sunday" in claim_text:
        score += 10

    # Rule 5: Previous claims (Mock)
    # Assuming clean history for this demo
    
    return {"fraud_score": score}

def settlement_computation(state: AgentState):
    """Calculates final settlement amount."""
    print("\n--- Node: Settlement Computation ---")
    damage = state["damage_cost"]
    fraud_score = state["fraud_score"]
    
    if fraud_score > 70:
        return {"settlement_estimate": 0, "approval_status": "Rejected (High Fraud Risk)"}
    
    # Depreciation (Mock logic based on vehicle age implied or fixed)
    depreciation_rate = 0.10 # 10% for 1-2 years
    deductible = 1000
    
    settlement = damage * (1 - depreciation_rate) - deductible
    if settlement < 0: settlement = 0
    
    status = "Approved"
    if state["missing_docs"]:
        status = "Conditional (Pending Docs)"
        # Note: In our flow, we loop for missing docs, so if we reach here, 
        # it might mean we accepted them or forced a decision.
        # But the loop logic handles the 'return to user' part. 
        # If we are here, we assume we proceed.
        
    return {"settlement_estimate": settlement, "approval_status": status}

def final_report_builder(state: AgentState):
    """Generates the final adjudication report."""
    print("\n--- Node: Final Report Builder ---")
    report = f"""
    *** INSURANCE CLAIM ADJUDICATION REPORT ***
    -------------------------------------------
    Incident Type: {state['parsed_fields'].get('incident_type')}
    Damage Cost: RS {state['damage_cost']}
    
    Policy Clauses Applied:
    {chr(10).join(['- ' + c for c in state['policy_clauses']])}
    
    Fraud Score: {state['fraud_score']}/100
    
    Settlement Calculation:
    - Base Amount: {state['damage_cost']}
    - Depreciation: 10%
    - Deductible: RS 1000
    - Final Estimate: RS {state['settlement_estimate']}
    
    Status: {state['approval_status']}
    
    Missing Documents: {state['missing_docs'] if state['missing_docs'] else 'None'}
    -------------------------------------------
    """
    return {"final_report": report}

# --- 4. Workflow Definition ---

def should_continue(state: AgentState):
    """Decides whether to loop back for missing docs or proceed."""
    missing = state.get("missing_docs", [])
    iteration = state.get("iteration_count", 0)
    
    if missing and iteration < 1: # Limit loops to 1 for demo
        print(f"!!! Missing Documents: {missing}. Requesting from user... (Simulated)")
        return "loop_back"
    return "proceed"

def input_simulator(state: AgentState):
    """Simulates user providing missing docs."""
    print("\n--- Node: Input Simulator (User Action) ---")
    # In a real app, this would pause or wait for input.
    # Here we simulate adding the missing docs.
    missing = state["missing_docs"]
    current_docs = state["parsed_fields"]["submitted_docs"]
    current_docs.extend(missing) # Simulate user uploading all missing docs
    
    new_text = state["claim_text"] + " . Added documents: " + ", ".join(missing)
    
    return {
        "parsed_fields": {**state["parsed_fields"], "submitted_docs": current_docs},
        "claim_text": new_text,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "missing_docs": [] # Clear missing docs
    }

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_claim_details", parse_claim_details)
    workflow.add_node("retrieve_rules_rag", retrieve_rules_rag)
    workflow.add_node("check_document_completeness", check_document_completeness)
    workflow.add_node("input_simulator", input_simulator)
    workflow.add_node("fraud_detection_agent", fraud_detection_agent)
    workflow.add_node("settlement_computation", settlement_computation)
    workflow.add_node("final_report_builder", final_report_builder)
    
    workflow.set_entry_point("parse_claim_details")
    
    workflow.add_edge("parse_claim_details", "retrieve_rules_rag")
    workflow.add_edge("retrieve_rules_rag", "check_document_completeness")
    
    workflow.add_conditional_edges(
        "check_document_completeness",
        should_continue,
        {
            "loop_back": "input_simulator",
            "proceed": "fraud_detection_agent"
        }
    )
    
    workflow.add_edge("input_simulator", "check_document_completeness") # Re-check
    workflow.add_edge("fraud_detection_agent", "settlement_computation")
    workflow.add_edge("settlement_computation", "final_report_builder")
    workflow.add_edge("final_report_builder", END)
    
    return workflow.compile()

# --- 5. Execution ---

if __name__ == "__main__":
    # Initialize RAG
    try:
        vectorstore = setup_rag_system()
    except Exception as e:
        print(f"RAG Setup failed: {e}")
        vectorstore = None

    # Sample Claim
    # Scenario: Accident, missing license initially
    sample_claim = """
    I had an accident with my car on Sunday. 
    The front bumper is damaged. Repair estimate is RS 45,000.
    I have attached my Policy Document and RC Copy.
    """
    
    initial_state = {
        "claim_text": sample_claim,
        "parsed_fields": {},
        "policy_clauses": [],
        "damage_cost": 0.0,
        "missing_docs": [],
        "fraud_score": 0,
        "approval_status": "",
        "settlement_estimate": 0.0,
        "final_report": "",
        "iteration_count": 0
    }
    
    print("Starting Insurance Claim Adjudication System...")
    app = build_graph()
    
    final_state = app.invoke(initial_state)
    
    print("\n\n" + "="*50)
    print("FINAL OUTPUT")
    print("="*50)
    print(final_state["final_report"])
    print("="*50)
