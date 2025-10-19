import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

# LangChain & Google GenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

# ==============================
# ENV + EVENT LOOP FIX
# ==============================
load_dotenv()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ==============================
# LOAD DOCUMENTS
# ==============================
file_path = r"C:\Users\Nomesh Chandrakar\Downloads\science college.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# ==============================
# EMBEDDINGS + VECTOR STORE
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2",  
    model_kwargs = {'device': 'cpu'},  
    encode_kwargs = {'normalize_embeddings': False} )


vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)

# ==============================
# SAFETY + LLM SETUP
# ==============================
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("gemini_key"),
    temperature=0.3,
    safety_settings=safety_settings
)

# ==============================
# PROMPT TEMPLATE
# ==============================
prompt_template = """
## Safety and Respect Come First!

You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

* Harassment or Bullying
* Hate Speech
* Violence or Harm
* Misinformation and Falsehoods

If the user request violates these guidelines, reply with:
"I'm here to assist with safe and respectful interactions. Your query goes against my guidelines. Let's try something different that promotes a positive and inclusive environment."

## Answering User Question:

Context: \n 
Question1: \n {how many seat in bcs cs}
Answer: match the seats number and give the answer and if there is link for that suggest a link also

Question2: \n {what is the fees for bsc cs}
Answer:  return fees for specific course

## if student ask any question that is not matching exact words use synonyms of that word to match and give answer

## if there are links avaible for details then after answering the question give the links also

## if they ask some question that we dont have answer suggest them to visit official website ans give the mail website link
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ==============================
# RETRIEVER
# ==============================
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    llm=chat_model
)

# ==============================
# FINAL QA CHAIN
# ==============================
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,                       # use Gemini chat model
    retriever=retriever_from_llm,         # ✅ must include retriever
    chain_type="stuff"                    # simplest chain type
)

# ==============================
# STREAMLIT UI
# ==============================
st.title("NareshIT Chatbot (RAG with Gemini)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask something about NareshIT...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run RAG QA
    response = qa_chain.invoke({"query": user_input})
    answer = response["result"]

    # Save bot reply
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
