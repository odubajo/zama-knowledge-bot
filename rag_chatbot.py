import streamlit as st
import chromadb
import google.generativeai as genai
from typing import List, Dict
from chromadb.utils import embedding_functions

class ZamaRAGChatbot:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.genai_client = None

    @st.cache_resource
    def initialize_chromadb(_self, db_path: str, collection_name: str):
        try:
            chroma_client = chromadb.PersistentClient(path=db_path)
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            collection = chroma_client.get_or_create_collection(
                name=collection_name, 
                embedding_function=embedding_func
            )
            return chroma_client, collection, True
        except Exception as e:
            st.error(f"Error connecting to ChromaDB: {str(e)}")
            return None, None, False

    @st.cache_resource
    def initialize_gemini(_self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            genai_client = genai.GenerativeModel('gemini-2.0-flash')
            return genai_client, True
        except Exception as e:
            st.error(f"Error initializing Gemini API: {str(e)}")
            return None, False

    def setup_connections(self, db_path: str, collection_name: str, api_key: str):
        self.chroma_client, self.collection, chroma_success = self.initialize_chromadb(db_path, collection_name)
        self.genai_client, gemini_success = self.initialize_gemini(api_key)
        return chroma_success and gemini_success

    @st.cache_data(ttl=300)
    def retrieve_relevant_docs(_self, query: str, n_results: int, collection_hash: str = "") -> List[Dict]:
        try:
            collection = st.session_state.get('collection')
            if not collection:
                return []
                
            results = collection.query(query_texts=[query], n_results=n_results)
            
            docs = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    docs.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    })
            return docs
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []

    def generate_response(self, query: str, context_docs: List[Dict], chat_history: List[Dict]) -> str:
        try:
            context = "\n\n".join([doc['content'] for doc in context_docs])
            
            history_string = ""
            for message in chat_history:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    history_string += f"User: {content}\n\n"
                else:
                    history_string += f"Zama Assistant: {content}\n\n"

            prompt = f"""You are an educative and conversational AI assistant specialized in Zama and Fully Homomorphic Encryption.
            Your task is to answer user questions based on the provided context only.
            
            Be friendly, and engaging. Explain complex topics clearly and provide examples when appropriate.
            
            Conversation History:
            {history_string}
            
            Context:
            {context}

            User Question: {query}
            
            Response:"""

            response = self.genai_client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"