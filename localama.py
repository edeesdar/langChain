# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

from langsmith import traceable
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import os


langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# if not openai_api_key:
#     st.error("Error: Missing OpenAI API Key. Check your .env file.")

if not langchain_api_key:
    st.warning("Warning: Missing LangChain API Key.")

# os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
print("LangChain API Key:", os.getenv("LANGCHAIN_API_KEY"))
print("LangChain Tracking:", os.getenv("LANGCHAIN_TRACKING_V2"))


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
# llm = Ollama(model="llama3.2")
llm = OllamaLLM(model="llama3.2")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser 



# Process input
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
