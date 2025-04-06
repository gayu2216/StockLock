import os
import streamlit as st
import yfinance as yf
import pdfplumber
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import base64
import gtts
from io import BytesIO

# Set page title and layout
st.set_page_config(page_title="RAG Assistant with Voice", layout="wide")

st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #FFDBED, #ffffff);
            padding: 40px;  /* Increased padding */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# App title and description
# App title and description
st.markdown("<h1 style='text-align: center;'>Voice-Enabled RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about your documents or get stock prices. You can type or use voice input!</p>", unsafe_allow_html=True)

# Initialize session states
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'gemini' not in st.session_state:
    st.session_state.gemini = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'submitted_query' not in st.session_state:
    st.session_state.submitted_query = ""

# Function to create an HTML audio player with the given text
def text_to_speech_player(text):
    tts = gtts.gTTS(text=text, lang="en", slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_bytes = fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay="true" controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
    return audio_tag

# Simplified Gemini interface class
class TinyGemini:
    def __init__(self, model_name="gemini-1.5-pro"):
        try:
            self.model = genai.GenerativeModel(model_name,
                                         generation_config={
                                             "max_output_tokens": 100,
                                             "temperature": 0.1,
                                             "top_p": 0.95
                                         })
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            self.model = None

    def generate(self, prompt):
        if self.model:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                st.error(f"Error generating content: {e}")
                return None
        else:
            return "Gemini model not initialized."

# Load PDFs efficiently
def load_books_minimal(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                    text = "".join(page.extract_text() for page in pdf.pages)
                    docs.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                st.warning(f"Error loading {filename}: {str(e)}")
    return docs

# Create smaller chunks
def create_small_chunks(docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Create vectorstore
def create_minimal_vectorstore(chunks):
    # Clean up old DB if it exists
    if os.path.exists("chroma_db"):
        import shutil
        shutil.rmtree("chroma_db")
        
    embeddings = FastEmbedEmbeddings()
    return Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

# Get stock data
def get_stock_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")['Close'][0]
    except:
        return None

# Voice to text function
def voice_to_text():
    r = sr.Recognizer()
    
    with st.spinner("Listening... Speak now"):
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_audio:
            st.info("Recording... Speak clearly into your microphone")
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=15)
            st.success("Recording complete! Processing speech...")
            try:
                text = r.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.error("Could not understand audio")
                return None
            except sr.RequestError as e:
                st.error(f"Speech service error: {e}")
                return None

# Handle form submission
def handle_query_submit():
    query = st.session_state.query_input
    st.session_state.submitted_query = query
    st.session_state.query_input = ""  # Clear the input field

# Centralized content area
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # API Key input (hardcoded)
    api_key = "AIzaSyA2EP7PqgtNGbbt96OLMbTolNAplzKjK7k"
    
    # Center the Run button within col2
    left_spacer, center_col, right_spacer = st.columns([1, 1, 1])
    with center_col:
        # Load PDFs button
        if st.button("Run"):
            if api_key:
                genai.configure(api_key=api_key)
                if st.session_state.gemini is None:
                    st.session_state.gemini = TinyGemini()
                
                pdf_folder = "books"
                
                if os.path.exists(pdf_folder):
                    with st.spinner("Loading documents..."):
                        docs = load_books_minimal(pdf_folder)
                        chunks = create_small_chunks(docs)
                        st.session_state.vectorstore = create_minimal_vectorstore(chunks)
                        st.session_state.documents_loaded = True
                        st.success(f"Loaded {len(docs)} documents and created {len(chunks)} chunks!")
                else:
                    st.error(f"Folder not found: {pdf_folder}")
            else:
                st.error("API key not set. Please set the API key in the code.")
    
    if st.session_state.documents_loaded and st.session_state.gemini:
        # Voice input button
        if st.button("🎤 Voice Input"):
            query = voice_to_text()
            if query:
                st.session_state.submitted_query = query
                st.info(f"You said: {query}")
        
        # Text input with form for better control
        with st.form(key="query_form"):
            st.text_input("Enter your question:", key="query_input")
            submit_button = st.form_submit_button(label="Ask")
            if submit_button:
                handle_query_submit()
        
        # Process the submitted query
        if st.session_state.submitted_query:
            query = st.session_state.submitted_query
            
            # Display a spinner while processing
            with st.spinner("Processing your question..."):
                if "stock price" in query.lower():
                    ticker = query.split(" ")[-1].upper()
                    price = get_stock_price(ticker)
                    if price:
                        response = f"{ticker}: ${price:.2f}"
                    else:
                        response = f"Error getting {ticker} price"
                else:
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                        docs = retriever.get_relevant_documents(query)
                        context = "\n".join(d.page_content[:250] for d in docs)
                        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer briefly:"
                        response = st.session_state.gemini.generate(prompt)
                    except Exception as e:
                        response = f"Error: {str(e)}"
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": query, 
                    "answer": response,
                    "audio": text_to_speech_player(response)
                })
                
                # Clear the submitted query to prepare for the next interaction
                st.session_state.submitted_query = ""

        # Display chat history
        st.markdown("### Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Question {i+1}:** {chat['question']}")
                st.markdown(f"**Answer {i+1}:** {chat['answer']}")
                st.markdown(chat['audio'], unsafe_allow_html=True)
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
                
        # Clear chat history button
        if st.button("Clear Chat History") and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.experimental_rerun()

# Add necessary requirements to requirements.txt
if not os.path.exists("requirements.txt"):
    with open("requirements.txt", "w") as f:
        f.write("streamlit\nyfinance\npdfplumber\ngoogle-generativeai\nlangchain\nlangchain-community\nlangchain-text-splitters\nfastembedapi\nspeech_recognition\ngtts\n")