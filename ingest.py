from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  
# from sentence_transformers import SentenceTransformer
  
DATA_FILE = "data/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}

def vector_db():
    loader = PyPDFLoader(DATA_FILE)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500, 
                chunk_overlap =50,
                length_function = len)

    texts = text_splitter.split_documents(docs)

    hf_embeddings = HuggingFaceEmbeddings(
                model_name = model_id,
                model_kwargs = model_kwargs)
                
    db = FAISS.from_documents(texts, 
                              hf_embeddings)

    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    vector_db()

