# ---- Streamlit Frontend ----
import streamlit as st
import requests

st.title("LLM Query Agent Web App")
query_text = st.text_input("Enter your query:")

if st.button("Submit"):
    if query_text:
        try:
            # Send the query to the FastAPI endpoint
            res = requests.post("http://127.0.0.1:8001/query", json={"query": query_text})
            if res.status_code == 200:
                st.write("Agent Response:")
                st.write(res.json()["response"])
            else:
                st.error(f"API Error: {res.status_code}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")