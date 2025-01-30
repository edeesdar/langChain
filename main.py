from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

if not openai_api_key:
    st.error("Error: Missing OpenAI API Key. Check your .env file.")

if not langchain_api_key:
    st.warning("Warning: Missing LangChain API Key.")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACKING_V2"] = "true"

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title("LangChain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want")

# Initialize LangChain components
llm = ChatOpenAI(model="gpt-3.5-turbo") 
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Process input
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
