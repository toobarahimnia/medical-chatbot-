from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
import chainlit as cl

DB_FAISS_PATH = "vectorstore/db_faiss"

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}


custome_prompt_template="""
Based on the following information try to answer the user's questions. 
If you are not sure what to say, don't make up an answer, 
just say I don't know the answer.
Context:{context}
Question:{question}
"""

# retrieval QA chain
def retrieval_qa_chain(llm, db):
    qa_chain = RetrievalQA.from_chain_type(llm = llm,
        chain_type="stuff",
        return_source_documents=True,
        retriever=db.as_retriever(search_kwargs={"k":2}), # retrieving top 2 docs
        chain_type_kwargs={
            "prompt":PromptTemplate(
            template=custome_prompt_template,
            input_variables=["context", "question"],
            ),
        },
    )
    return qa_chain

# load the model
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama", # not llama2
        config={
            "max_new_tokens":512,
            "temperature":0.5,
        },
    ) 

    return llm

# QA model function
def QA_model():
    llm=load_llm()

    embeddings = HuggingFaceEmbeddings(
        model_name = model_id,
        model_kwargs = model_kwargs)

    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings)
    
    qa=retrieval_qa_chain(llm, db)
    
    return qa


@cl.on_chat_start
async def on_chat_start():
    chain=QA_model()
    msg=cl.Message(content="Loading...")
    await msg.send()
    msg.content="Hello and welcome to Medical Encyclopedia! How can I assist?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain=cl.user_session.get("chain")
    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True 
    res=await chain.acall(message.content, callbacks=[cb])
    answer=res["result"]
    source_docs=res["source_documents"]

    if source_docs:
        answer += f"\nSources: " + str(source_docs)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

# to run, type: chainlit run model.py -w
