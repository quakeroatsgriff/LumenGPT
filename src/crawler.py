import requests
import os
import openai
import pandas as pd
import numpy as np
import re
import dill as pickle

from numpy.linalg import norm
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

def main():
    url = "https://content.one.lumenlearning.com/introductiontopsychology/"
    # url = "https://content.one.lumenlearning.com/introductiontopsychology/chapter/1-1-5-learn-it-the-history-of-psychology-gestalt-psychology/"
    # url = "https://docs.python.org/3.9/"

    loader = RecursiveUrlLoader( url=url, max_depth=1, extractor=lambda x: BeautifulSoup( x, "html.parser" ).text )
    pickle.dump( loader, open( './pickles/loader.pkl', 'wb' ) )
    print("pickeled loader")

    docs = loader.load()
    pickle.dump( docs, open( './pickles/docs.pkl', 'wb' ) )
    print("pickeled docs")

    for doc in docs:
        doc_title = doc.metadata.get( 'title' )
        print( "Examining", doc.metadata.get('source') )
        # Skip over files with no title (XML, CSS, etc.)
        if not doc_title:
            # print("SKIPPING", doc.metadata.get('source'))
            continue

        # Clean up file name
        doc_title = re.sub( " ", "_", doc_title )
        doc_title = re.sub( "/", "_", doc_title )
        filepath = "./textbook_pages_OG/"+ doc_title

        with open( filepath, "w+" ) as doc_file:
            doc_file.write( doc.page_content )
            # print("printed",doc_title)

if __name__ == "__main__":
    main()