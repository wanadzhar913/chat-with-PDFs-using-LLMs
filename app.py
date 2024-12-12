import os
import logging
from typing import List, Union

import streamlit as st
from pypdf import PdfReader

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s %(asctime)s %(message)s"
)
logger = logging.getLogger(__name__)

def get_pdf_text(pdf_docs: Union[str, list]) -> List[str]:
    """
    initialise a variable which takes an empty string and subsequently
    apend the pages of the pdf to this variable.

    ### Arguments
    - `pdf_docs`: This can take in either a string (file path) or
    a list of `BytesIO` objects (due to Streamlit's `UploadFile` being
    a subclass of `BytesIO`). Reference: https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/file_util.py#L33C1-L61C28

    ### Return
    A string of extracted text from the PDF.
    """
    text = ""

    if isinstance(pdf_docs, list):
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    else:
        # we instead read the file from a directory
        pdf_reader = PdfReader(pdf_docs)
        for page in pdf_reader.pages:
            text += page.extract_text()
 
    return text

def get_text_chunks(text: str) -> List[str]:
    """
    Here we use the `RecursiveCharacterTextSplitter` class
    to split our document into smaller text chunks. Reference: https://python.langchain.com/docs/how_to/recursive_text_splitter/

    ### Arguments
    - `text`: The text (string) to be split into 800 character chunks
    with 200 character overlap.

    ### Return
    A list of text chunks each with a size of 800 characters.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap=200,
        length_function = len,
        is_separator_regex = False,
    )

    chunks = text_splitter.split_text(text)

    return chunks

def get_vectorstore(text_chunks: List[str]):
    """Function to call our vector store."""
    # create embeddings before loading into a vector store/knowledge base
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    # we'll use FAISS as our vector store.
    vector_store = FAISS.from_texts(text_chunks, embeddings)

    return vector_store

def get_conversation_chain(vectorstore):
    """
    This is a function to set up the conversation chain.
    This will be the mechanism through which we allow the chatbot to use
    the vector store when trying to answer questions.    
    """
    llm = ChatOpenAI(
        model = "gpt-4o-mini",
        temperature = 0.25,
        max_tokens = 1024,
        max_retries = 2,
    )

    retriever = vectorstore.as_retriever()

    memory = ConversationBufferMemory(
        llm=llm,
        input_key="question",
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    return chain

def handle_user_input(user_question):
    """
    Function that receives the users query, calls the LLM and vector stores
    and prints the output conversation.
    """
    config = {"configurable": {"session_id": "any"}}
    response = st.session_state.conversation.invoke({'question': user_question, 'chat_history': st.session_state.chat_history}, config)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            user = st.chat_message(name="User", avatar="💃")
            user.write(message.content)
        else:
            assistant = st.chat_message(name="J.A.A.F.A.R.", avatar="🤖")
            assistant.write(message.content)

def main():
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon="🤓",
    )

    if os.getenv("OPENAI_API_KEY") == None:
        os.environ["OPENAI_API_KEY"] = st.secrets["api_keys"]["OPENAI_API_KEY"]
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click `Process`", accept_multiple_files=True
        )

        if st.button("Process"):
            
            if pdf_docs == []:
                st.write('Please upload a document first!')
            else:
                with st.spinner("Processing files ..."):
                    logger.info("User is processing a PDF.")
                    
                    st.session_state.pdf_upload = True

                    # get raw pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # segment raw pdf text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # load text chunks into a vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Chat with multiple PDFs 📚")

    with st.container(border=True):
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            if 'pdf_upload' not in st.session_state:
                assistant = st.chat_message(name="J.A.A.F.A.R.", avatar="🤖")
                assistant.write('Please upload a document and click `Process` first before asking anything.')
            else:
                handle_user_input(user_question)

if __name__ == '__main__':
    main()
