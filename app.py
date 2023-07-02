import streamlit as st
from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 800,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):

    # create embeddings before loading into a vector store
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vectorstore):
    
    llm = ChatOpenAI(openai_api_key=st.secrets["api keys"]["OPEN_API_KEY"],
                     temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    user = st.chat_message(name="User", avatar="💃")
    assistant = st.chat_message(name="J.A.A.F.A.R.", avatar="🤖")

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            assistant.write(message.content)
        else:
            user.write(message.content)

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon="🤓")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDFs 📚")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click `Process`", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing files ..."):
                # get raw pdf text
                raw_text = get_pdf_text(pdf_docs)

                # segment raw pdf text into chunks
                text_chunks = get_text_chunks(raw_text)

                # load text chunks into a vector store
                vectorstore= get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()