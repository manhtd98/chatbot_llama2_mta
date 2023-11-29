from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from config import CONFIG
from loguru import logger


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=CONFIG.CUSTOMER_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model=CONFIG.MODEL_NAME, model_type="llama", max_new_tokens=512, temperature=0.5
    )
    return llm


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG.EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(CONFIG.DB_FAISS_PATH, embeddings)
    logger.info("Success load vector store")
    llm = load_llm()
    logger.info("Success load LLM model")
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    logger.info("Loaded QA retrieval")

    return qa


# output function
chain = qa_bot()


def final_result(query):
    response = chain({"query": query})
    return response


# chainlit code
@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)
    logger.info("Success init chat app")


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
