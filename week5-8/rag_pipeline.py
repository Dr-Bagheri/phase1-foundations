from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI



def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )



def retrieve_context(query, k=3):
    db = load_vector_store()
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])


# prompt template
RAG_PROMPT = """
You are a helpful assistant. Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""



def generate_answer(query):
    context = retrieve_context(query)

    prompt = RAG_PROMPT.format(
        context=context,
        question=query
    )

    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output[0].content[0].text



if __name__ == "__main__":
    query = "What are the main conclusions of the documents?"

    answer = generate_answer(query)

    print("\n Answer:\n")
    print(answer)