import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents(folder_path):
    documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
    
    return documents



def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)



def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )



def build_vector_store(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def main():
    folder_path = "./pdfs"  
    
    print(" Loading documents...")
    documents = load_documents(folder_path)

    print(f"Loaded {len(documents)} pages")

    print(" Splitting into chunks...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print(" Generating embeddings...")
    embeddings = create_embeddings()

    print(" Building FAISS index...")
    vector_store = build_vector_store(chunks, embeddings)


    vector_store.save_local("faiss_index")

    print(" Index saved to ./faiss_index")


if __name__ == "__main__":
    main()