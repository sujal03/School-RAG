# import streamlit as st
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# import os

# # Configuration
# CHROMA_PERSIST_DIR = "chroma_db"

# # Initialize components
# @st.cache_resource
# def init_components():
#     # Initialize Gemini embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vectordb = Chroma(
#         persist_directory=CHROMA_PERSIST_DIR,
#         embedding_function=embeddings
#     )
#     # Initialize Gemini chat LLM (using the gemini-pro model)
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", streaming=True, temperature=0.5)
#     return vectordb, llm

# # Styling
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #0E1117;
#         color: #FFFFFF;
#     }
#     .stChatInput input {
#         background-color: #1E1E1E !important;
#         color: #FFFFFF !important;
#         border: 1px solid #3A3A3A !important;
#     }
#     .stChatMessage {
#         background-color: #1E1E1E !important;
#         border: 1px solid #3A3A3A !important;
#         color: #FFFFFF !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # App title
# st.title("📚 Document Chat Assistant")

# # Check if ChromaDB exists
# if not os.path.exists(CHROMA_PERSIST_DIR):
#     st.error("⚠️ No indexed documents found! Please run the indexer script first.")
#     st.stop()

# # Initialize components
# vectordb, llm = init_components()

# # Chat prompt template
# PROMPT_TEMPLATE = """
# You are a knowledgeable assistant helping with document-related questions.
# Base your answer ONLY on the following context. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
# Do not make up or infer any information that is not directly stated in the context.

# Context: {context}

# Question: {question}

# ### Answer Format:
#  - Provide a clear, structured response.
#  - Give response in bullet point if needed and in proper format.
# """
# prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# if question := st.chat_input("Ask me about your documents..."):
#     # Display user message
#     with st.chat_message("user"):
#         st.markdown(question)
#     st.session_state.messages.append({"role": "user", "content": question})

#     # Generate response
#     with st.chat_message("assistant"):
#         # Create a placeholder for the streaming response
#         response_placeholder = st.empty()
        
#         with st.spinner("Thinking..."):
#             # Search for relevant documents
#             docs = vectordb.similarity_search_with_score(question, k=4)
#             # Filter docs with a lower score (more relevant) than the threshold
#             relevant_docs = [doc[0] for doc in docs if doc[1] < 0.5]
#             context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
#             # Combine the prompt and LLM into a chain
#             chain = prompt | llm
#             response_chunks = []
            
#             # Stream the response and extract text from each chunk
#             for chunk in chain.stream({
#                 "context": context,
#                 "question": question
#             }):
#                 response_chunks.append(chunk)
#                 # Join the content from each chunk rather than the chunk objects themselves
#                 current_response = "".join(chunk.content for chunk in response_chunks)
#                 response_placeholder.markdown(current_response)
            
#             final_response = "".join(chunk.content for chunk in response_chunks)
#             st.session_state.messages.append({"role": "assistant", "content": final_response})




























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
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

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
            context = "\n\n".join(doc.page_content for doc in relevant_docs) if relevant_docs else "No relevant context available."

            if not relevant_docs:
                response_text = "⚠️ No relevant documents found. Try rephrasing your question."
                response_placeholder.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.stop()

        # Generate response from LLM
        with st.spinner("🤖 Generating answer..."):
            chain = prompt | llm
            response_chunks = []
            
            for chunk in chain.stream({"context": context, "question": question}):
                response_chunks.append(chunk)
                current_response = "".join(chunk.content for chunk in response_chunks)
                response_placeholder.markdown(current_response)
            
            final_response = "".join(chunk.content for chunk in response_chunks)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
