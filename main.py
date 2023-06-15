# Initialize libs
# pip install openai
# pip install pypdf
# pip install chroma
# pip install tiktoken

import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb

# imports
import config
import utils

# initialize
os.environ["OPENAI_API_KEY"] = config.openai_apikey
embeddings = OpenAIEmbeddings()

#setup memory object
chat_memory = ConversationBufferMemory(human_prefix="human",ai_prefix="AI", memory_key="chat_history", return_messages=True)
# Db directory
persist_directory = 'dbtxt'
#initial DB
vectordb = Chroma(persist_directory=persist_directory)

# init method
def init(db):
    prompt = initChain(db)
    return prompt


# function to split the docs and prepare for storing

def prepareDocForUpload(uploaded_doc):
    print(type(uploaded_doc))
    load_documents = utils.textfrompdf(uploaded_doc)
    pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    return pdf_text_splitter.split_text(text=load_documents)


def clearDB():
    vectordb.delete_collection()


def loadTextoDB(text):
    # init and load data
    vectordb = Chroma.from_texts(texts=text,embedding=embeddings,persist_directory=persist_directory)
    vectordb.persist()
    print("done persist text!!!!!")
    return vectordb


# set LLMChains
def initChain(db):
    return ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5),
                                                 db.as_retriever(search_type="mmr"), memory=chat_memory, verbose=True)

def chatPrompt(prompt, query):
    result = prompt({"question": query})
    return result
