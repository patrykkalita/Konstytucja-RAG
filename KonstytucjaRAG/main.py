# Zaimportowanie potrzebnych bibliotek
import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


# Funkcja, która odpowiada za wczytanie pliku pdf pod wskazaną ścieżką i pobiera z niego tekst
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Mapowanie pliku pdf
pdf_mapping = {
    'Konstytucja': 'Konstytucja.pdf'
}

# Załadowanie zmiennych środowiskowych z pliku .env
load_dotenv()


# Główna funkcja aplikacji
def main():
    # Ustawienie tytułu i opisu
    st.title("Konstytucja RP")
    with st.sidebar:
        st.title('🤗💬 Chatbot Konstytucji RP')
        st.markdown('''
        ## Chatbot
        Wybierz plik aby rozpocząć
        ''')


    # Wybór z listy pliku pdf do wczytania
    custom_names = list(pdf_mapping.keys())
    selected_custom_name = st.sidebar.selectbox('Wybierz plik', ['', *custom_names])
    selected_actual_name = pdf_mapping.get(selected_custom_name)

    if selected_actual_name:
        file_path = os.path.join(selected_actual_name)

        # Obsługa wczytania pliku pdf
        try:
            text = read_pdf(file_path)
            st.info("Zawartość dokumentu jest ukryta. Zadaj pytanie na temat Konstytucji RP.")
        except FileNotFoundError:
            st.error(f"Nie znaleziono pliku: {file_path}")
            return
        except Exception as e:
            st.error(f"Wystąpił błąd podczas wczytywania pliku PDF: {e}")
            return

        # Inicjalizacja obiektu do podziału tekstu na fragmenty
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Podział wczytanego pdf na fragmenty
        documents = text_splitter.split_text(text=text)

        # Utworzenie wektorów osadzeń tekstu i zbudowanie wektorowej bazy danych do przechowywania słów znajdujących się w dokumencie
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        # Zapisanie przetworzonych danych w sesji Streamlit
        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Zapisanie wektorowej bazy danych za pomocą modułu pickle
        pickle_folder = "Pickle"
        if not os.path.exists(pickle_folder):
            os.mkdir(pickle_folder)

        pickle_file_path = os.path.join(pickle_folder, f"{selected_custom_name}.pkl")

        if not os.path.exists(pickle_file_path):
            with open(pickle_file_path, "wb") as f:
                pickle.dump(vectorstore, f)

        # Szablon odpowiedzi dla modelu językowego OpenAI
        system_template = """Use the following pieces of context to answer the users question about constitution of Poland. 
                If you cannot find the answer from the pieces of context and question is not about polish constitution, just say that you don't know, don't try to make up an answer.
                ----------------
                {context}"""

        # Utworzenie szablonu dla wiadomości
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        # Inicjalizacja modelu językowego OpenAI
        llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        # Inicjalizacja łańcucha konwersacji (Langchain), wskazujemy jakiego modelu używamy, źródła danych oraz przekazujemy customowy szablon odpowiedzi
        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = vectorstore.as_retriever(),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt":qa_prompt}
        )

        # Inicjalizacja chatu Streamlit oraz wyświetlenie dotychczasowo wysłanych wiadomości
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Pole w którym możemy zadać pytanie
        if prompt := st.chat_input("Zadaj pytanie na temat "f'{selected_custom_name}'"?"):
            # Dodanie do sesji wiadomości wysłanych przez użytkownika
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Wyświetlenie wszystkich wiadomości wysłanych w sesji
            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            print(prompt)

            # Wygenerowanie odpowiedzi dla użytkownika przez model językowy
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            print(full_response)

            # Dodanie odpowiedzi modelu do sesji
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()