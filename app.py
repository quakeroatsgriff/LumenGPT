import os
import openai
from sys import stdin

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from src.database import load_db
from dotenv import load_dotenv

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_embedding(text: str, model: str ="text-embedding-ada-002") -> list:
    """ Get embeddings from text-embedding-ada model
    Args: strings of text chunks
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def retrieve_from_db( vectordb: Chroma, prompt: str = "This is a test prompt." ):
    """ Finds related embeddings from the database given a prompt
    Args: Vector DB and prompt string

    Return related documents from DB
    """
    # retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = vectordb.similarity_search( prompt )
    return docs

def assemble_prompt( embedded_txt: str, question_prompt: str = None ):
    """ Assembles a text prompt by reading in separate text files for different componets

    Returns: Text string of prompt to ask the LLM
    """

    background_txt = """You are a chatbot for Lumen Learning that is tasked with answering
        questions asked by students that are reading the introductory psychology textbook.
        Students are able to type into a text box and ask their question to you. In a short while,
        you will see a question that the student asked, denoted in triple backticks (``` text ```).
        Before being prompted with this, a separate embedding model (OpenAI Ada 2) was prompted with
        the same question that the student is asking here. Ada 2 will provide 4 related text
        chunks from the data it was trained as well as the source where those text chunks
        come from. The responses from that model will be helpful and provide some
        context for the correct answer. The text from Ada 2 will also be provided in triple
        backticks. The text will be in the form of a Python dictionary, with two lists being
        "page_content" and "source". Each element in these lists will be associated with each other
        in the same order.
    """
    instructions_txt = """Your job is to:
        1. Parse out the "page_content" and "source" text and match them together from the Ada 2 model.
        This will be important to keep track of not only which source is associated with which page content,
        but also the ordering of each element. Each of these pairs will be used for the foot notes later.

        2. Analyze the response from the previous model. Identify the topics mentioned
        and what exactly is being explained.
        3. Record the sources for each response from the Ada 2 model.
        4. Analyze the student's question and think of how you would respond to it with no
        background information.
        5. Combine your response with the responses from the Ada 2 model.
        6. Next to each claim, cite where the claim comes from using the sources listed by the
        Ada 2 model. You may use footnotes for this.
    """
    output_txt = """Your output should look like one short paragraph that answers the students
    question. After every claim, cite it with a footnote ([4] for example). At the end of the
    paragraph, create a list of footnotes for each source. Make sure to write out each source's location"""

    question_txt = question_prompt
    # Just in case no question was supplied, this is a default test question
    if question_txt == None:
        question_txt = """Who is Sigmund Freud?"""

    example_scenario_txt = """An example question is if the student asks 'Who is Sigmund Freud?"
        From the Ada 2 model, it states that Freud was 'was an Austrian neurologist who was fascinated
        by patients suffering from “hysteria” and neurosis.'. Your response may include that text from Ada 2
        with a footnote at the end of it looking like [1]. At the end of your response paragraph, there will
        be a bulleted list of footnotes that contains the footnote number and the name of the source. In this
        case, it would look like [2] 'Early_Psychology:_Learn_It_5-Psychoanalytic_Theory_-_Intro_Psych'
    """
    example_output_txt = """Sigmund Freud was an Austrian neurologist who was fascinated by patients
        suffering from "hysteria" and neurosis [4]. He is considered one of the most influential and
        well-known figures in the history of psychology [2]. Freud's interest in hysteria and neurosis
        led him to develop psychoanalytic theory, which explores the unconscious mind and the role of
        unconscious desires and conflicts in shaping human behavior [2]. His theories have had a
        significant impact on the field of psychology and continue to be studied and debated today [2].

        * [2] 'Intro_Psych'
        * [4] 'Early_Psychology:_Learn_It_5—Psychoanalytic_Theory_–_Intro_Psych'
    """
    prompt = f"""
        {background_txt}
        {instructions_txt}
        {output_txt}
        {example_scenario_txt}
        Full Example output: {example_output_txt}
        Support information from Ada 2: ```{embedded_txt}```
        Student's question: ```{question_txt}```
    """
    return prompt

def main():
    # Get env variables and objects
    load_dotenv()
    openai.api_key = os.getenv( "OPENAI_API_KEY" )

    embedding = OpenAIEmbeddings()
    vectordb = load_db( persist_directory = "./database", embedding = embedding )
    users_question = "Who is Sigmund Freud?"
    print("Ask the bot a prompt:")
    users_question = stdin.read()
    print( "Input Prompt:\n", users_question )
    docs = retrieve_from_db( vectordb = vectordb, prompt = users_question )
    doc_content = {"source": [], "page_content": [] }
    for i, doc in enumerate( docs ):
        print( doc.page_content )
        print( doc.metadata )
        print()
        doc_content["source"].append( doc.metadata.get("source") )
        doc_content["page_content"].append( doc.page_content )
    print("Generating answer by GPT")
    prompt = assemble_prompt( doc_content, question_prompt = users_question )
    response = get_completion(prompt)
    print(response)

if __name__ == "__main__":
    main()