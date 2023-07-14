import streamlit as st
# import os for apikey setup
import os
from langchain.llms import OpenAI

# import pdf document loaders
from langchain.document_loaders import PyPDFLoader

# import chroma as the vector store
from langchain.vectorstores import Chroma
# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
pages = loader.load_and_split()

# load documents into vector database
store = Chroma.from_documents(pages, collection_name='annualreport')

os.environ['OPENAI_API_KEY'] = 'insert key here'

# instanciate OpenAI LLM
llm = OpenAI(temperature=0.9)

# text box for user input
prompt = st.text_input('Enter prompt:')

if prompt:
    response = llm(prompt)
    st.write(response)

    # with streamlit expander
    with st.expander('Document Similarity Search'):
        # find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # write out the first
        st.write(search[0][0].page_content)