import os
import openai
import re

from numpy.linalg import norm
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader

# def load_documents( filepath_dir: str ) -> list[str]:
#     """ Reads in a series of documents from a directory
#     Args: filepath string to document directory to pull from
#     Returns: list of the content on each document page
#     """
#     dir_files = os.listdir( filepath_dir )
#     doc_list = []
#     for file in dir_files:
#         # Add "/" at end of filepath string if needed
#         if filepath_dir[-1:] != "/":
#             filepath_dir += "/"
#         filepath = filepath_dir + file
#         with open( filepath, "r+" ) as file_obj:
#             # print( "Loading", filepath, "into the list" )
#             doc_list.append( file_obj.read() )
#     return doc_list

def get_embedding(text: str, model: str ="text-embedding-ada-002") -> list:
    """ Get embeddings from text-embedding-ada model
    Args: strings of text chunks
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def load_db( persist_directory = "./database/", embedding = any ):
    """ Loads vector database from disk"""
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

def join_db(lower: int = 0, upper: int = 1, db_dir: str = "database") -> Chroma:
    """ Combines multiple smaller Chroma databases together. Also saves the database
    to disk.

    Args:
        - Lower bound, upper bound for number of databases
        - Root database directory
    Return: A bigger Chroma database object
    """
    embedding = OpenAIEmbeddings
    root_db = load_db( persist_directory = db_dir + "/d0", embedding = embedding )

    db_dir_list = os.listdir( db_dir )
    # Remove root database since we don't need to join it
    db_dir_list.remove( "d0" )
    for db_file in db_dir_list:
        # Get iterated database from disk
        db = load_db( persist_directory = db_dir + "/" + db_file, embedding = embedding )
        db_data = db._collection.get( include = ['documents','metadatas','embeddings'] )
        # Add database information to the root database
        root_db._collection.add(
            embeddings = db_data['embeddings'],
            documents = db_data['documents'],
            metadatas = db_data['metadatas'],
            ids = db_data['ids']
        )
    root_db._persist_directory = db_dir + "/db_combined"
    root_db.persist()
    return root_db

def create_db():
    """ Creates the vector database and saves it to disk.
    Returns: The database object
    """

    # Get env variables and objects
    load_dotenv()
    openai.api_key = os.getenv( "OPENAI_API_KEY" )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 200,
        length_function = len,
    )
    # Get list of the content on each document page
    loader = DirectoryLoader( 'textbook_pages_new' )
    doc_list = loader.load()
    # doc_list = load_documents( './textbook_pages_new/' )

    # Clean up source path in metadata
    for doc in doc_list:
        match = re.search(r'[^/]+$', doc.metadata['source'])
        if match:
            doc.metadata['source'] = match.group()
        doc.metadata['source'] = re.sub( r'_', ' ', doc.metadata['source'] )

    # Split up text into tokens of small chunk sizes
    texts = text_splitter.split_documents(doc_list)

    # for i, text in enumerate( texts[0:5] ):
        # print( i, text )

    # create new list with all text chunks
    text_chunks=[]

    for text in texts:
        text_chunks.append(text.page_content)

    # print(len(text_chunks))
    # print(text_chunks)

    persist_directory = 'database'
    # OpenAI embeddings
    embedding = OpenAIEmbeddings()

    # Create IDs for each text chunk
    id_range = range( 0, len( texts ) + 1 )
    ids = [ str(num) for num in id_range ]
    # Create databases of 10,000 text chunks each
    # for i in range( 0, ( len(texts) // 10000 ) + 1 ):
    #     persist_directory = f'database/d{i}'
    #     print(f"working on {0+(10000*i)}:{10000+(10000*i)}")
    #     texts_sliced = texts[0+(10000*i):10000+(10000*i)]
    #     ids_sliced = ids[0+(10000*i):10000+(10000*i)]
    #     # Create database from text, NOT from file
    #     vectordb = Chroma.from_documents(documents = texts_sliced,
    #         embedding = embedding,
    #         persist_directory = persist_directory,
    #         ids = ids_sliced
    #     )
    #     # Persist the db to disk (save it to file)
    #     vectordb.persist()
    persist_directory = f'database'
    # Create database from text, NOT from file
    vectordb = Chroma.from_documents(documents = texts,
        embedding = embedding,
        persist_directory = persist_directory,
        ids = ids
    )
    # Persist the db to disk (save it to file)
    vectordb.persist()
    return vectordb


def main():
    """ This just creates or joins the vector database. """
    create_db()
    # join_db()

if __name__ == "__main__":
    main()