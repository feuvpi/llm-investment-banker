import streamlit as st
# import os for apikey setup
import os
from langchain.llms import OpenAI

# import pdf document loaders
from langchain.document_loaders import PyPDFLoader

# import chroma as the vector store
from lamngchain.vectorstores import Chormaa




os.environ['OPENAI_API_KEY'] = 'insert key here'

# instanciate OpenAI LLM
llm = OpenAI(temperature=0.9)

# text box for user input
prompt = st.text_input('Enter prompt:')

if prompt:
    response = llm(prompt)
    st.write(response)