class CONFIG:
    DB_FAISS_PATH = "vectorstore/db_faiss"
    MODEL_NAME = "TheBloke/vietnamese-llama2-7B-40GB-GGUF"
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

    CUSTOMER_PROMPT_TEMPLATE = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng.
     Nếu không biết câu trả lời, bạn chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.

    Ngữ cảnh: {context}
    Câu hỏi: {question}
    Câu trả lời:
    """
