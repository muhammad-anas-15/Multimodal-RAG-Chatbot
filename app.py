import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import os
from datetime import datetime

# Import LangChain and Google Generative AI components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Extracts text from uploaded PDF files.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Splits the extracted text into smaller chunks for processing.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)  # Split the big text into many small chunks.
    return chunks

# Converts text chunks into embeddings and saves them locally using FAISS.
def get_vector_store(text_chunks, api_key):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Initialize Google Embedding Model
    # 1) Converts every text chunk into numbers.
    # 2) Stores them in FAISS (a fast search database)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save locally

# Model Training for answering
def get_conversational_chain(api_key):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.
    Answer the question based ONLY on the following context.

    ### Guidelines for Answering:
    1. **Format your answer using Markdown.**
    2. Use **bullet points** for list items or steps.
    3. Use **bold text** for key terms or important headings.
    4. Keep paragraphs short and readable.
    5. If the answer is not in the context, say: "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}
    """)
    # Sends the PDF text (context) and the user’s question into the prompt and then to the Gemini(LLM) to get answer.
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | model
    )
    return chain

# Handles the user's question: loads the index, searches for similar content, and generates a response.
def user_input(user_question, api_key):
    if not os.path.exists("faiss_index"):  # Check if the vector store exists
        st.error("Please process the PDF files first.")
        return
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load the vector store with dangerous deserialization allowed (forcely open this even if old file)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)  # Get the chain and run it
    # Combines those PDF parts into one text, that will be send to AI for answer
    context = "\n\n".join([doc.page_content for doc in docs])
    # Ask AI --> sends the question + context
    response = chain.invoke({
        "context": context,
        "question": user_question
    })
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.conversation_history.append({
        "question": user_question,
        "answer": response.content,
        "timestamp": timestamp
    })

# 3. Main UI of Application
def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="✨", layout="wide")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # CSS Styling
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', sans-serif;
            }
            /* 1. Main Background */
            .stApp {
                background-color: #f8f9fa;
                background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
                background-size: 20px 20px;
            }
            /* 2. Sidebar Styling */
            [data-testid="stSidebar"] {
                background-color: #0f172a;
                border-right: 1px solid #1e293b;
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                color: #f1f5f9 !important;
            }
            [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
                color: #94a3b8 !important;
            }
            /* Sidebar Buttons */
            .stButton > button {
                width: 100%;
                border-radius: 12px;
                height: 50px;
                font-weight: 600;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            /* Primary Action Button (Process) */
            div[data-testid="stSidebar"] .stButton > button {
                background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
                border: none;
                color: white;
                box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
            }
            div[data-testid="stSidebar"] .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
            }
            /* File Uploader */
            [data-testid="stFileUploader"] {
                background-color: #1e293b;
                border-radius: 12px;
                padding: 1rem;
            }
            /* Main Chat Container */
            .block-container {
                max-width: 1000px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            /* Chat Animations */
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .chat-row {
                display: flex;
                margin-bottom: 1.5rem;
                animation: slideIn 0.4s ease-out;
            }
            .user-row { justify-content: flex-end; }
            .bot-row { justify-content: flex-start; }
            .chat-bubble {
                padding: 1.25rem 1.5rem;
                border-radius: 1.5rem;
                max-width: 80%;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                position: relative;
                line-height: 1.6;
            }
            .user-bubble {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                border-bottom-right-radius: 0.25rem;
            }
            .bot-bubble {
                background: white;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-bottom-left-radius: 0.25rem;
            }
            .avatar {
                width: 40px; height: 40px;
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                margin: 0 1rem;
                font-size: 1.2rem;
                flex-shrink: 0;
            }
            .user-avatar { background: #e0e7ff; color: #4338ca; }
            .bot-avatar { background: #f1f5f9; color: #0f172a; }
            /* Input Field Styling */
            .stTextInput input {
                border-radius: 2rem;
                padding: 15px 25px;
                border: 1px solid #cbd5e1;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
                min-height: 30px;
                font-size: 1rem
            }
            .stTextInput input:focus {
                border-color: #6366f1;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
            }
            /* Hide Default Elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Chat Input field Enlarge */
            div[data-testid="stHorizontalBlock"] div[data-testid="stTextInput"] {
                height: 68px;
                display: flex;
                align-items: center;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stTextInput"] > div {
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stTextInput"] input {
                height: 60px !important;
                min-height: 60px !important;
                padding: 0 28px !important;
                font-size: 1.05rem !important;
                line-height: normal !important;
                display: flex;
                align-items: center;
                border-radius: 2rem;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stTextInput"] input::placeholder {
                line-height: normal;
            }
            div[data-testid="stHorizontalBlock"] {
                align-items: center !important;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
                height: 60px !important;
                min-height: 60px !important;
                padding: 0 40px !important;
                border-radius: 14px !important;
                font-size: 1.1rem !important;
                font-weight: 600;
                background: linear-gradient(135deg, rgb(236,72,153), rgb(59,130,246)) !important;
                border: none !important;
                color: white !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-shadow: 0 6px 20px rgba(236,72,153,0.35);
                transition: all 0.25s ease;
            }
            div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 25px rgba(236,72,153,0.45);
            }
            div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
                background-color: white !important;
                border-radius: 8px !important;
                padding: 0.4rem 1rem !important;
                margin-bottom: 0.3rem !important;
                display: flex;
                align-items: center;
                justify-content: space-between;
                color: #1e293b !important;
                min-height: 36px !important;
            }
            div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] button {
                background-color: white !important;
                color: #ef4444 !important; /* red cross icon */
                border-radius: 50% !important;
                width: 22px;
                height: 22px;
                padding: 0;
                display: flex;
                align-items: center;
                justify-content: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Sidebar Content
    # -----------------------------
    with st.sidebar:
        st.markdown("## 🤖 **AI Assistant**")
        st.caption("Powered by Google Gemini & LangChain")

        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔑 **Access**")
        api_key = st.text_input("Google API Key", type="password", placeholder="Enter key...")

        st.markdown("### 📂 **Knowledge Base**")
        pdf_docs = st.file_uploader("Upload Documents (PDF)", accept_multiple_files=True)

        if st.button("⚡ Analyze Documents"):
            if not api_key:
                st.error("Please provide an API Key.")
            elif not pdf_docs:
                st.error("Please upload documents.")
            else:
                with st.spinner("🧠 Processing content..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Analysis Complete!")
            st.markdown("---")

        if st.session_state.conversation_history:
            df = pd.DataFrame(st.session_state.conversation_history)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Chat", data=csv, file_name='chat_history.csv', mime='text/csv')

    # -----------------------------
    # Main Chat Section
    # -----------------------------
    st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
            <h1 style='color: #1e293b; font-weight: 800; font-size: 3rem; letter-spacing: -1px;'>
                Document<span style='color: #6366f1;'>AI</span>
            </h1>
            <p style='color: #64748b; font-size: 1.1rem;'>Upload PDFs and chat with your documents instantly.</p>
        </div>
    """, unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        if not st.session_state.conversation_history:
            st.markdown("""
                <div style='background: white; padding: 3rem; border-radius: 1rem; text-align: center; border: 2px dashed #cbd5e1; opacity: 0.7;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>👋</div>
                    <h3 style='color: #1e293b;'>Welcome!</h3>
                    <p style='color: #64748b;'>Upload a PDF on the left to get started.</p>
                </div>
            """, unsafe_allow_html=True)

        for chat in st.session_state.conversation_history:
            # User Message
            st.markdown(f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">
                        {chat['question']}
                        <div style="font-size: 0.7rem; margin-top: 5px; opacity: 0.8; text-align: right;">{chat['timestamp']}</div>
                    </div>
                    <div class="avatar user-avatar">👤</div>
                </div>
            """, unsafe_allow_html=True)

            # Bot Message
            st.markdown(f"""
                <div class="chat-row bot-row">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="chat-bubble bot-bubble">
                        {chat['answer']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # -----------------------------
    # Input Area
    # -----------------------------
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1])
    with col1:
        user_question = st.text_input("Message", label_visibility="hidden", placeholder="Ask a question about your documents...")
    with col2:
        # Styled button
        st.markdown("""
            <style>
                div[data-testid="column"]:nth-of-type(2) button {
                    height: 48px;
                    border-radius: 50%;
                    width: 48px !important;
                    padding: 0;
                    float: right;
                    background: #1e293b;
                    color: white;
                    border: none;
                }
                div[data-testid="column"]:nth-of-type(2) button:hover {
                    background: #334155;
                }
            </style>
        """, unsafe_allow_html=True)
        # Trigger loading state
        if st.button("➤", type="primary"):
            if user_question and api_key:
                with st.spinner("🧠 Thinking..."):
                    user_input(user_question, api_key)
                st.session_state["user_input"] = ""
                st.rerun()

if __name__ == "__main__":
    main()
