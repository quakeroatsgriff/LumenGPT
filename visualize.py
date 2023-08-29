from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from chromaviz.visualize import visualize_collection
from src.database import load_db

embedding = OpenAIEmbeddings
vectordb = load_db( persist_directory = "./database", embedding = embedding )
visualize_collection(vectordb._collection)

