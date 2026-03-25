import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI



client = OpenAI()



def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store



def retrieve_context(vector_store, query, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])


# prompt
def build_prompt(context, question):
    return f"""
You are a helpful assistant. Answer the question using ONLY the context below.

If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""



def generate_answer(prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output[0].content[0].text




st.set_page_config(page_title="Chat with your PDF", layout="wide")

st.title(" Chat with your PDF (RAG)")


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    if "vector_store" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.vector_store = process_pdf(uploaded_file)
        st.success("PDF processed and indexed!")


    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Thinking..."):
            context = retrieve_context(st.session_state.vector_store, query)
            prompt = build_prompt(context, query)
            answer = generate_answer(prompt)

        st.markdown("###  Answer")
        st.write(answer)


        with st.expander(" Retrieved Context"):
            st.write(context)