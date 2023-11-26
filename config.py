class CONFIG:
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    MODEL_NAME = "TheBloke/vietnamese-llama2-7B-40GB-GGUF"
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    
    CUSTOMER_PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """