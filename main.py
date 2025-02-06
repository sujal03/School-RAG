import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Manually set the API key from the environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY is missing. Please check your .env file.")
    st.stop()

# Configuration
CHROMA_PERSIST_DIR = "chroma_db"

@st.cache_resource
def init_components():
    """Initialize vector database and Gemini LLM."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectordb = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, streaming=True, temperature=0.5)
    return vectordb, llm

# Custom Styling
st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stTextInput > div > div > input {
        background-color: #1E1E1E !important; color: white !important;
    }
    .stChatMessage { background-color: #1E1E1E !important; color: white !important; }
    .stSpinner { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# App Title
st.title("📚 AI-Powered Document Chat Assistant")

# Check for indexed documents
if not os.path.exists(CHROMA_PERSIST_DIR):
    st.error("⚠️ No indexed documents found! Please run the indexer script first.")
    st.stop()

# Initialize ChromaDB and LLM
vectordb, llm = init_components()

# Chat Prompt Template
PROMPT_TEMPLATE = """
You are an AI assistant helping users extract information from documents.
Use only the provided context to answer questions.

Context:
{context}

User Question:
{question}

### Answer Guidelines:
- Be clear and concise.
- Use points for structured answers.
- Avoid assumptions or fabrications.
"""

FALLBACK_PROMPT = """
You are an AI assistant helping users with their queries.
Since there is no relevant document-based context, provide a **general answer** based on your knowledge.

User Question:
{question}

### Answer Guidelines:
- Keep the response **brief** and **to the point**.
- Provide factual, general knowledge.
- Avoid unnecessary elaboration.
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
fallback_prompt = ChatPromptTemplate.from_template(FALLBACK_PROMPT)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if question := st.chat_input("Ask me about your documents..."):
    # Show user input
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("🔍 Searching documents..."):
            # Retrieve relevant documents
            docs = vectordb.similarity_search_with_score(question, k=5)
            relevant_docs = [doc[0] for doc in docs if doc[1] < 0.5]
            context = "\n\n".join(doc.page_content for doc in relevant_docs) if relevant_docs else ""

        # Determine which prompt to use
        if context:
            final_prompt = prompt
            prompt_data = {"context": context, "question": question}
        else:
            final_prompt = fallback_prompt
            prompt_data = {"question": question}

        # Generate response from LLM
        with st.spinner("🤖 Generating answer..."):
            chain = final_prompt | llm
            response_chunks = []
            
            for chunk in chain.stream(prompt_data):
                response_chunks.append(chunk)
                current_response = "".join(chunk.content for chunk in response_chunks)
                response_placeholder.markdown(current_response)
            
            final_response = "".join(chunk.content for chunk in response_chunks)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
