[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_red.svg)](https://chat-with-pdfs-using-llms-kedxembo5i.streamlit.app/)

### **TLDR**

Simple PDF chatbot using LangChain, OpenAI, FAISS and Streamlit.

### **Sample User Interface**

!['Sample GUI'](images/Interface_v020723.png)

### **Getting Started**

#### 1.0 For deploying to Steamlit Cloud

This should be pretty straightforward. Link your GitHub and Streamlit account and Streamlit's
Community Cloud should take care of the CI/CD.

```{bash}
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

streamlit run app.py
```

#### 2.0 Docker Container
If you'd like to deploy via Docker, we can get started with the below. Be sure to
set your `OPENAI_API_KEY` environment variable below!

```{bash}
docker build -t simplepdfchatbot .
docker run -e OPENAI_API_KEY="" -p 8501:8501 simplepdfchatbot
```

### **Resources**

- https://www.youtube.com/watch?v=dXxQ0LR-3Hg
- https://medium.com/@manasabrao2020/building-a-rag-pdf-chat-application-using-streamlit-and-langchain-3d30dc225bad
- https://docs.streamlit.io/library/api-reference/chat/st.chat_message
- https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html
