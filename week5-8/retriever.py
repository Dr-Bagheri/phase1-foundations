from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



def load_vector_store(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db



def retrieve_top_k(query, k=3):
    db = load_vector_store()

    results = db.similarity_search(query, k=k)

    return results



if __name__ == "__main__":
    query = "What are the key findings of the document?"

    docs = retrieve_top_k(query)

    print("\n Top Results:\n")

    for i, doc in enumerate(docs):
        print(f"--- Result {i+1} ---")
        print(doc.page_content)
        print("\n")