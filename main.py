from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from handle_project import handle_project_file

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str
    project: str

@app.post('/ask')
async def ask(query: Query):
    txt_path = handle_project_file(query.project)
    loader = TextLoader(txt_path)
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    documents = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mistral")
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./db")
    llm = ChatOllama(model="mistral")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
    )
    
    result = qa_chain({"question": query.question})
    return {"data": result["answer"]}
