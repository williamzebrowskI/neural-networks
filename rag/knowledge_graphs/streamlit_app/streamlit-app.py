#This python code will create an interactive Chatbot to talk to documents.
import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from streamlit_option_menu import option_menu
#Let's integrate langsmith
from dotenv import load_dotenv, find_dotenv
from langsmith import Client
#Import related to KnowledgeGraph
from py2neo import Graph
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

from knowledge_graph import RAG_Graph


#Create temporary folder location for document storage
TMP_DIR = Path(__file__).resolve().parent.parent.joinpath('data','tmp')


header = st.container()

def streamlit_ui():

    with st.sidebar:
        choice = option_menu('Navigation',["Home",'Simple RAG','RAG with Neo4J'])

    if choice == 'Home':
        st.title("RAG tutorial using multiple techniques")

    elif choice == 'Simple RAG':
        with header:
            st.title('Simple RAG with vector')  
            st.write("""This is a simple RAG process where user will upload a document then the document
                     will go through RecursiveCharacterSplitter and embedd in FAISS DB""")
            
            source_docs = st.file_uploader(label ="Upload a document", type=['pdf'], accept_multiple_files=True)
            if not source_docs:
                st.warning('Please upload a document')
            else:
                RAG(source_docs)
    
    elif choice == 'RAG with Neo4J':
        with header:
            st.title('RAG wih Neo4J')
            st.write("""This is RAG approach with Neo4J knowledge graph. After uploading document -> click on Load Graph.
                     Knowledge graph will display in chatbot. Responses of user queries will be fetched using hybrid search approach""")
            RAG_Neo4j()
           
            
def RAG(docs):
    #load the document
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())

    
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()

    #Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    #Vector and embeddings
    DB_FAISS_PATH = 'vectorestore/faiss'
    embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2',
                                         model_kwargs={'device':'cpu'})
    db = FAISS.from_documents(text,embedding)
    db.save_local(DB_FAISS_PATH)

    #Setup LLM, Fetch base url from LM Studio
    # llm = ChatOpenAI(base_url="http://localhost:1234/v1",api_key='lm-studio')
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o") 

    #Build a conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True
    )
    chat_history = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        #Add user message to chat history
        st.session_state.messages.append({"role":"user","context":prompt})

        response = f"Echo: {prompt}"
        #Display assistant response in chat message container
        response = qa_chain({'question':prompt,'chat_history':chat_history})

        with st.chat_message("assistant"):
            st.markdown(response['answer'])

        st.session_state.messages.append({'role':"assistant", "content":response})
        chat_history.append({prompt,response['answer']})

def RAG_Neo4j():
    rag_graph = RAG_Graph()
    choice = option_menu('Options',["Upload document",'Graph(Skip document upload)'])
    
    if choice == 'Upload document':
        source_docs = st.file_uploader(label="Upload document", type=['docx'],accept_multiple_files=True)
        if not source_docs:
            st.warning("Please upload a document")
        else:
            rag_graph.create_graph(source_docs,TMP_DIR)
    else:
        show_graph()

    
    st.session_state.messages1 = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages1 =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt1 := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt1)
        #Add user message to chat history
        st.session_state.messages1.append({"role":"user","context":prompt1})

        response1 = f"Echo: {prompt1}"
        #Display assistant response in chat message container
        response1 = rag_graph.ask_question_chain(prompt1)

        with st.chat_message("assistant"):
            st.markdown(response1)

        st.session_state.messages1.append({'role':"assistant", "content":response1})

def RAG_Neo4j1(docs,TMP_DIR):
    rag_graph = RAG_Graph()
    #rag_graph.create_graph(docs,TMP_DIR)
    show_graph()

    chat_history1 = []
    st.session_state.messages1 = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages1 =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt1 := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt1)
        #Add user message to chat history
        st.session_state.messages1.append({"role":"user","context":prompt1})

        response1 = f"Echo: {prompt1}"
        #Display assistant response in chat message container
        #response = qa_chain({'question':prompt,'chat_history':chat_history})
        response1 = rag_graph.ask_question_chain(prompt1)

        with st.chat_message("assistant"):
            st.markdown(response1['answer'])

        st.session_state.messages1.append({'role':"assistant", "content":response1})
        chat_history1.append({prompt1,response1['answer']})



def show_graph():
    st.title("Neo4j Graph Visualization")

    #user input for Neo4J credential
    uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
    user = st.text_input("Neo4j username", "neo4j")
    password = st.text_input("Neo4j password", type="password")

    #Create a load graph button
    if st.button("Load Graph"):
        try:
            data = get_graph_data(uri,user,password)
            G = create_networkx_graph(data)
            visualize_graph(G)

            HtmlFile = open("graph.html", "r", encoding="utf-8")
            source_code = HtmlFile.read()
            components.html(source_code,height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error loading page:  {e}")

def get_graph_data(uri,user,password):
    graph = Graph(uri,auth=(user,password))
    query = """
    MATCH (n)-[r]->(m)
    RETURN n,r,m
    LIMIT 100
    """

    data = graph.run(query).data()
    return data

def create_networkx_graph(data):
    G = nx.DiGraph()
    for record in data:
        n = record['n']
        m = record['m']
        r = record['r']
        G.add_node(n['id'], label=n['name'])
        G.add_node(m['id'], label=m['name'])
        G.add_edge(n['id'], m['id'], label=r['type'])
    return G

def visualize_graph(G):
    net = Network(notebook=True)
    net.from_nx(G)
    net.show("graph.html")


streamlit_ui()