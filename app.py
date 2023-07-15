import streamlit as st
# import os for apikey setup
import os
from langchain.llms import OpenAI

# import pdf document loaders
from langchain.document_loaders import PyPDFLoader

# import vector storestuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# import chroma as the vector store
from langchain.vectorstores import Chroma
# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')
pages = loader.load_and_split()

os.environ['OPENAI_API_KEY'] = 'insert key here'

# instanciate OpenAI LLM
llm = OpenAI(temperature=0.9)

# load documents into vector database
store = Chroma.from_documents(pages, collection_name='annualreport')

# create vectorestore info object
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a company annual report as a pdf - apple",
    vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# add toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# text box for user input
prompt = st.text_input('Enter prompt:')

if prompt:
    # pass prompt to llm
    response = llm(prompt)
    # swap out the raw llm for a document agent
    response = agent_executor.run(prompt)
    st.write(response)

    # with streamlit expander
    with st.expander('Document Similarity Search'):
        # find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # write out the first
        st.write(search[0][0].page_content)