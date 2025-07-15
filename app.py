import streamlit as st
import os
from pdf_processor import PDFProcessor
from ai_assistant import AIAssistant
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

st.set_page_config(
    page_title="Physik-Curriculum Assistent",
    page_icon="📚",
    layout="wide"
)

def load_existing_vectorstore():
    """Lädt bestehende Vektor-Datenbank"""
    if os.path.exists("./chroma_db"):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        return vectorstore
    return None

def main():
    st.title("📚 Physik-Curriculum Assistent")
    st.markdown("Stellen Sie Fragen zum Fach-Curriculum Physik Schleswig-Holstein")
    
    # Sidebar für PDF-Upload
    st.sidebar.header("PDF-Dokument verarbeiten")
    
    uploaded_file = st.sidebar.file_uploader(
        "Laden Sie Ihr PDF-Dokument hoch",
        type="pdf"
    )
    
    if uploaded_file is not None:
        # Speichere hochgeladene Datei temporär
        with open("temp_curriculum.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if st.sidebar.button("PDF verarbeiten"):
            with st.spinner("Verarbeite PDF-Dokument..."):
                processor = PDFProcessor()
                vectorstore = processor.process_pdf("temp_curriculum.pdf")
                st.sidebar.success("PDF erfolgreich verarbeitet!")
                st.session_state.vectorstore = vectorstore
    
    # Lade bestehende Vektor-Datenbank
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = load_existing_vectorstore()
    
    # Haupt-Chat-Interface
    if st.session_state.vectorstore is not None:
        st.header("💬 Chat mit dem Curriculum")
        
        # Initialisiere Chat-Verlauf
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Zeige Chat-Verlauf
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat-Input
        if prompt := st.chat_input("Stellen Sie eine Frage zum Curriculum..."):
            # Füge Benutzernachricht hinzu
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generiere Antwort
            with st.chat_message("assistant"):
                with st.spinner("Denke nach..."):
                    assistant = AIAssistant(st.session_state.vectorstore)
                    response = assistant.ask_question(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Bitte laden Sie zuerst ein PDF-Dokument hoch und verarbeiten Sie es.")
        
        # Beispiel-Fragen anzeigen
        st.subheader("Beispiel-Fragen:")
        st.markdown("""
        - Welche Lernziele sind für die Klasse 7 definiert?
        - Wie ist das Thema Mechanik strukturiert?
        - Welche Experimente werden für die Optik empfohlen?
        - Wie wird das Thema Energie behandelt?
        """)

if __name__ == "__main__":
    main()
