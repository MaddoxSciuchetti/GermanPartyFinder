import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pathlib
from chat import ask_deepseek
from config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    EMBEDDINGS_MODEL,
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_FILE_TYPES
)

class DocumentProcessor:
    """Handles document processing and vector store operations."""
    
    @staticmethod
    def get_pdf_text(pdf_file):
        """Extract text from a PDF file."""
        text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def get_txt_text(txt_file):
        """Extract text from a TXT file."""
        return txt_file.getvalue().decode("utf-8")

    @staticmethod
    def process_documents(docs):
        """Process multiple documents and combine their text."""
        text = ""
        for doc in docs:
            if doc.type == "application/pdf":
                text += DocumentProcessor.get_pdf_text(doc)
            elif doc.type == "text/plain":
                text += DocumentProcessor.get_txt_text(doc)
        return text

    @staticmethod
    def get_text_chunks(text):
        """Split text into manageable chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        return splitter.split_text(text)

    @staticmethod
    def create_vector_store(text_chunks):
        """Create and save a FAISS vector store."""
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        pathlib.Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)
        return vector_store

class PoliticalAnalyzer:
    """Handles political analysis and recommendations."""
    
    @staticmethod
    def get_party_recommendation(profile: str, language: str) -> str:
        """Analyze user profile and recommend political parties."""
        prompt = f"""Basierend auf dem folgenden Profil, führen Sie eine detaillierte Analyse durch, 
        welche deutsche Partei am besten zu der Person passt. Berücksichtigen Sie dabei alle relevanten 
        Parteien (SPD, CDU/CSU, Grüne, FDP, LINKE, AfD).

        Persönliches Profil:
        {profile}

        Bitte strukturieren Sie Ihre Analyse wie folgt:
        1. Zusammenfassung der wichtigsten Werte und Prioritäten aus dem Profil
        2. Analyse der Übereinstimmungen mit den verschiedenen Parteien
        3. Detaillierte Begründung für die am besten passende(n) Partei(en)
        4. Mögliche Vorbehalte oder zu bedenkende Aspekte
        5. Alternative Optionen mit Begründung

        Beachten Sie:
        - Aktuelle politische Positionen der Parteien
        - Konkrete Übereinstimmungen bei wichtigen Themen
        - Mögliche Konfliktpunkte
        - Lokale und bundesweite Perspektiven

        Antworten Sie ausführlich und ausschließlich auf Deutsch."""
        
        return ask_deepseek(prompt, language=language)

    @staticmethod
    def answer_political_question(context: str, question: str, language: str) -> str:
        """Answer political questions based on provided context."""
        prompt = f"""Basierend auf folgendem Kontext, analysieren Sie die Frage aus der Perspektive 
        der deutschen Parteienlandschaft und aktueller politischer Entwicklungen.

        Kontext:
        {context}

        Frage:
        {question}

        Bitte berücksichtigen Sie:
        - Positionen aller relevanten Parteien
        - Aktuelle politische Entwicklungen
        - Historische Zusammenhänge
        - Gesellschaftliche Auswirkungen

        Strukturieren Sie Ihre Antwort klar und antworten Sie ausschließlich auf Deutsch."""
        
        return ask_deepseek(prompt, language=language)

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'language' not in st.session_state:
        st.session_state.language = DEFAULT_LANGUAGE

def create_political_qa_section(col, language):
    """Create the Political Q&A section."""
    with col:
        st.header("Politisches Analyse-System 🏛️")
        
        # Document upload section
        st.subheader("Dokumente hochladen")
        uploaded_files = st.file_uploader(
            "Laden Sie politische Dokumente hoch:",
            accept_multiple_files=True,
            type=SUPPORTED_FILE_TYPES
        )
        
        if uploaded_files:
            if st.button("Dokumente verarbeiten"):
                with st.spinner("Verarbeitung läuft..."):
                    text = DocumentProcessor.process_documents(uploaded_files)
                    chunks = DocumentProcessor.get_text_chunks(text)
                    DocumentProcessor.create_vector_store(chunks)
                    st.session_state.processed_docs = True
                    st.success("Dokumente erfolgreich verarbeitet!")

        st.markdown("### Politische Fragen stellen")
        question = st.text_input(
            "Ihre Frage:",
            placeholder="z.B.: Wie stehen die verschiedenen Parteien zum Klimaschutz?"
        )

        if question:
            if not st.session_state.processed_docs:
                st.info("📚 Bitte laden Sie zuerst Dokumente hoch!")
            else:
                with st.spinner("Analysiere..."):
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
                    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(question, k=3)
                    context = " ".join([doc.page_content for doc in docs])
                    
                    answer = PoliticalAnalyzer.answer_political_question(context, question, language)
                    st.markdown("### Analyse:")
                    st.markdown(answer)

def create_party_matcher_section(col, language):
    """Create the Party Matcher section."""
    with col:
        st.header("Partei-Finder 🎯")
        
        profile = st.text_area(
            "Beschreiben Sie Ihre politischen Ansichten und Werte:",
            height=300,
            placeholder="""Beschreiben Sie zum Beispiel:
- Ihre wichtigsten politischen Themen
- Ihre Werte und Überzeugungen
- Ihre Sicht zu aktuellen Entwicklungen
- Ihre Vorstellungen für die Zukunft Deutschlands
- Ihre Position zu Wirtschaft, Umwelt, Sozialem etc."""
        )
        
        if st.button("Parteien-Analyse starten"):
            if profile:
                with st.spinner("Führe Analyse durch..."):
                    recommendation = PoliticalAnalyzer.get_party_recommendation(profile, language)
                    st.markdown("### Ihre persönliche Parteien-Analyse")
                    st.markdown(recommendation)
            else:
                st.warning("Bitte geben Sie zuerst Ihre politischen Ansichten ein!")

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Politisches Analyse-System",
        page_icon="🏛️",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("Politisches Analyse-System 🏛️")
    st.markdown("""
    Willkommen beim Politischen Analyse-System! Dieses Tool hilft Ihnen:
    - Ihre politische Position in der deutschen Parteienlandschaft zu finden
    - Politische Dokumente und Positionen zu analysieren
    - Aktuelle politische Fragen zu beantworten
    """)
    
    # Create two equal columns
    left_col, right_col = st.columns(2)
    
    # Create both main sections
    create_party_matcher_section(left_col, DEFAULT_LANGUAGE)
    create_political_qa_section(right_col, DEFAULT_LANGUAGE)

if __name__ == "__main__":
    main()