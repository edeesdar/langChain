# Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langsmith import traceable
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set LangChain environment variables
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

if not langchain_api_key:
    st.warning("Warning: Missing LangChain API Key.")

# Ensure correct LangSmith tracing environment variables
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # Required for cloud tracing

print("LangChain API Key:", os.getenv("LANGCHAIN_API_KEY"))
print("LangChain Tracing:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LangChain Endpoint:", os.getenv("LANGCHAIN_ENDPOINT"))

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title("LangChain Demo with Ollama API")
input_text = st.text_input("Search the topic you want")

# Initialize LangChain components
llm = OllamaLLM(model="llama3.2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser 

# Enable tracing using LangSmith
@traceable
def run_chain(question):
    return chain.invoke({"question": question})

# Process input
if input_text:
    response = run_chain(input_text)  # Call the traced function
    st.write(response)
