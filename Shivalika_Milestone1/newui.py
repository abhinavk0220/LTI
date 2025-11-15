import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/ask/"

# Streamlit App UI
st.title("🛡️ SOC Analyst Assistant")

st.write("""
Welcome to the **Incident RAG Investigation System**.  
Enter your analyst ID and describe the security alert or question you want help with.
""")

# Input fields
user_id = st.text_input("🔐 Analyst ID")
query = st.text_area("📝 Describe the alert or ask a question")

# Submit button
if st.button("Ask"):
    if user_id and query:
        try:
            payload = {"user_id": user_id, "query": query}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer")

                st.subheader("✅ Assistant Response")

                if isinstance(answer, dict):
                    st.markdown(f"**🛠 Suggested Resolution:**\n{answer.get('resolution', 'N/A')}")
                    st.markdown("**📂 Similar Past Incidents:**")
                    for case in answer.get("similar_cases", []):
                        st.markdown(f"- {case}")
                    st.markdown(f"**📊 Threat Summary:**\n{answer.get('threat_summary', 'N/A')}")
                else:
                    st.write(answer)

            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ An error occurred: {e}")
    else:
        st.warning("Please enter both Analyst ID and a Question.")