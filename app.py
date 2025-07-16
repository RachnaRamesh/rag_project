import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


custom_prompt = PromptTemplate.from_template("""
Answer the question as detailed as possible from the provided context. 
If the answer is not in the context, respond with "answer is not available in the context". 
Don't make up answers.

Context: {context}
Question: {question}

Answer:
""")

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(BytesIO(pdf.read()))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(text_chunks, embedding=embeddings)
    store.save_local("faiss_index")

# Create conversational retrieval chain
def get_conversational_chain(vector_store):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=False
    )

# Main Streamlit app
def main():
    st.set_page_config("Chat with PDFs", layout="centered")

   
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #e8c3f3, #ffe6f0);
            font-family: 'Helvetica', sans-serif;
            color: #2c003e !important;
        }
        h1, h2, h3, .stTextInput label {
            color: #2c003e !important;
        }
        .stTextInput > div > div > input {
            background-color: #fff0f6;
            color: #2c003e;
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid #b96bb2;
        }
        .stButton button {
            background-color: #e1bee7;
            color: #2c003e;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #ce93d8;
            color: white;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #eec9f6, #f9d8ff);
            color: #2c003e;
            padding: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chat with Your PDFs 💬")

    # Sidebar file upload and processing
    with st.sidebar:
        st.header("📎 Upload Your PDFs")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)

                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    st.session_state.chain = get_conversational_chain(vector_store)
                    st.session_state.chat_history = []
                st.success("✅ Ready to chat with your PDFs!")
            else:
                st.warning("⚠️ Please upload at least one PDF.")

    if st.session_state.chain:
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your PDFs:", key="question_input")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_question:
            response = st.session_state.chain.invoke({"question": user_question})
            st.session_state.chat_history.append((user_question, response["answer"]))

    # Chat display area
    chat_placeholder = st.container()
    with chat_placeholder:
        for q, a in st.session_state.chat_history:
            st.markdown(f"""
                <div style='background-color:#f9e0f9;padding:12px;border-radius:12px;margin-bottom:10px;'>
                    <b>You:</b><br>{q}</div>
                <div style='background-color:#fff;padding:12px;border-radius:12px;margin-bottom:20px;border:1px solid #ccc;'>
                    <b>PDF Bot:</b><br>{a}</div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
