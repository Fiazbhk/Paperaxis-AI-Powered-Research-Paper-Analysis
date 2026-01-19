import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# -----------------------------
# ENV + MODELS
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=llm_model
)

embedder = HuggingFaceEmbeddings(
    model_name=embedding_model
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if "sections" not in st.session_state:
    st.session_state.sections = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -----------------------------
# 1. TITLE
# -----------------------------
st.set_page_config(layout="centered")

st.sidebar.title("❉ Paperaxis")

st.sidebar.markdown(
    """
**AI-powered Research Paper Analysis**

This project leverages AI and NLP to analyze research papers, generate summaries,
extract keywords, and visualize trends.

**Tech Stack:** Python, AI / ML, NLP, PDF Parsing, Streamlit
"""
)

st.sidebar.markdown("### Key Features")

st.sidebar.markdown(
    """
- Upload and analyze research PDFs
- Extract sections automatically from paper
- Generate summaries per selected section
- Chat with paper using AI
- Restart chat to clear history
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
[GitHub](https://github.com/Fiazbhk) | [LinkedIn](https://www.linkedin.com/in/fiazbhk/) | [LeetCode](https://leetcode.com/u/muhammadfiazbhk/)
"""
)

st.title("❉ Paperaxis")

# -----------------------------
# Horizontal Menu (Tabs)
# -----------------------------
selected_tab = option_menu(
    menu_title=None,
    options=["Get Summary", "Chat with Paper"],
    icons=["file-text", "chat"],
    orientation="horizontal",
    default_index=0
)

# -----------------------------
# GET SUMMARY TAB
# -----------------------------
if selected_tab == "Get Summary":
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_text = extract_text_from_pdf(file_path)
        st.session_state.full_text = extracted_text

        extracted_sections = extract_pdf_sections(full_text=extracted_text)
        refined_sections = refine_sections(extracted_sections, llm)
        section_with_content = split_sections_with_content(
            extracted_text,
            refined_sections
        )

        st.session_state.sections = section_with_content
        st.session_state.vector_db = None

        st.success("PDF processed successfully")

    if st.session_state.sections:
        st.subheader("Section Summary")

        section_names = list(st.session_state.sections.keys())
        selected_section = st.selectbox(
            "Select Section",
            section_names,
            key="selected_section"
        )

        if st.button("Generate Summary"):
            content = st.session_state.sections[st.session_state.selected_section]
            summary = generate_detailed_summary(content, llm)
            st.write(summary)

# -----------------------------
# CHAT TAB (Enhanced Chat UI)
# -----------------------------
if selected_tab == "Chat with Paper":
    st.subheader("Chat with Research Paper")

    if not st.session_state.full_text:
        st.warning("Upload a paper first in the Get Summary tab.")
    else:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat input
        user_message = st.chat_input("Ask a question...")

        # Display previous messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_message:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_message)

            # Generate response using your RAG chain
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if not st.session_state.vector_db:
                        st.session_state.vector_db = create_vector_db(
                            text=st.session_state.full_text,
                            embedder=embedder
                        )

                    chain = get_qa_chain(
                        vectordb=st.session_state.vector_db,
                        llm=llm
                    )

                    response = chain.invoke(user_message)["result"]
                    st.markdown(response)

            # Append messages to session state
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Restart button to clear chat history
        def clear_chat():
            st.session_state.messages = []

        st.button("Restart Chat", on_click=clear_chat)

