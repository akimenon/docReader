# Initialize libs
# pip install openai
# pip install pypdf
# pip install chroma
# pip install tiktoken
# pip install wikipedia

# imports
import config
import utils
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.document_loaders import WikipediaLoader
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
import chromadb

# initialize
os.environ["OPENAI_API_KEY"] = config.openai_apikey
embeddings = OpenAIEmbeddings()
wikisearch = WikipediaAPIWrapper()

#setup memory object
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#  chat_memory.input_key="question"
# chat_memory.output_key="answer"

llm=OpenAI(model_name="gpt-3.5-turbo",temperature=0)

# Db directory
persist_directory = 'dbtxt'
#initial DB
vectordb = Chroma(persist_directory=persist_directory)

# init method
def init(db):
    prompt = initChain(db)
    agent=initAgent(prompt)
    return agent


# function to split the docs and prepare for storing

def prepareDocForUpload(uploaded_doc):
    print(type(uploaded_doc))
    load_documents = utils.textfrompdf(uploaded_doc)
    pdf_text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],chunk_size=250, chunk_overlap=50)
    return pdf_text_splitter.split_text(text=load_documents)


def clearDB():
    print("delete all data")
    vectordb.delete_collection()


def loadTextoDB(text):
    # init and load data
    vectordb = Chroma.from_texts(texts=text,embedding=embeddings,persist_directory=persist_directory)
    vectordb.persist()
    print("done persist text!!!!!")
    return vectordb


# set LLMChains
def initChain(db):

    return ConversationalRetrievalChain.from_llm(llm,
                                                 db.as_retriever(search_type="mmr"), memory=chat_memory, verbose=True,return_source_documents=False)

def initAgent(qaprompt):
    tools = [
        Tool(
            name="MusicSearch",
            func=lambda query: qaprompt({"question": query}),
            description="For any questions on Music ask this tool first"
        )
        ,
        Tool(
            name="wikiSearch",
            func=wikisearch.run,
            description="Use this tool for generic searches on artist, events, world, and general knowledge"
        )
    ]
    agent_chain = initialize_agent(tools,llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                                   memory=chat_memory, max_iterations=19)


    #print(agent_chain.agent)
    return agent_chain


def chatPrompt(agent, query):
   response= agent(query)
   return response
