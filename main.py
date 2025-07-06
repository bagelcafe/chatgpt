__import__('pysqlite3')  # Ensure sqlite3 is imported to avoid import errors in some environments
import sys
sys.modules['sqlite3'] = sys.modules.pop['pysqlite3']


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

st.title("PDF Document Question Answering")
st.write("Ask questions about the content of the PDF document.")

# Ensure the OPENAI_API_KEY is set
#openai_api_key = os.getenv("OPENAI_API_KEY")

def pdf_to_document(upload_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, upload_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(upload_file.getvalue())

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    return pages

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", key="openai_api_key")

#upload file
uploaded_file = st.file_uploader("Choose a only PDF file", type="pdf")

button(username="bagelcafe", floating = True, width=221)

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    pass 

    st.write(f"Number of pages loaded: {len(pages)}")
    print(f"Number of pages loaded: {len(pages)}")
    print(f'Loaded pages: {pages[0].page_content[:100]}...')  # Display first 500 characters of the first page

    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
        )

    texts = text_split.split_documents(pages)


    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = Chroma.from_documents(texts, embeddings)

    st.header("Ask a question about the PDF document")
    question = st.text_input("Enter your question here", key="question_input")
    #question = "전기 피해와 관련 보상 알려줘"
    
    if st.button("Ask Question", on_click=lambda: st.session_state.update({"question": question})):
        with st.spinner("Wait for it to generate the answer..."):
            if not question:
                st.error("Please enter a question.")
                st.stop()
            llm = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key= openai_api_key, temperature=0)

            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever(), chain_type="stuff")
            result = qa_chain({"query": question})

            st.write("Retrieving documents...")
            st.write(result['result'])





