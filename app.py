__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from rag_chatbot import ZamaRAGChatbot
import config

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Your CSS styles remain here */
</style>
""", unsafe_allow_html=True)

def main():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ZamaRAGChatbot()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'system_initialized' not in st.session_state:
        with st.spinner("Initializing Zama educative assistant..."):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            absolute_db_path = os.path.join(script_dir, config.CHROMADB_PATH)
            
            success = st.session_state.chatbot.setup_connections(
                absolute_db_path, config.COLLECTION_NAME, config.GEMINI_API_KEY
            )
            
            if success:
                st.session_state.system_initialized = True
                st.session_state.collection = st.session_state.chatbot.collection
                st.session_state.collection_hash = f"{absolute_db_path}_{config.COLLECTION_NAME}"
            else:
                st.error("Failed to initialize system. Please check your configuration.")
                st.stop()
    
    st.image("zama.jpg", width=200)
    st.title(config.PAGE_TITLE)
    st.markdown("*Your intelligent guide to understanding Fully Homomorphic Encryption...*")
    st.markdown("---")

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Zama Assistant:</strong><br>{content}</div>', unsafe_allow_html=True)
            if "sources" in message:
                with st.expander("当 Source Documents"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f'<div class="source-docs"><strong>Source {i+1}:</strong><br>{doc["content"][:200]}...</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask me anything about Zama and FHE..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)

        with st.spinner("､Thinking..."):
            collection_hash = st.session_state.get('collection_hash', '')
            relevant_docs = st.session_state.chatbot.retrieve_relevant_docs(
                prompt, config.N_RESULTS, collection_hash
            )
            
            response = st.session_state.chatbot.generate_response(prompt, relevant_docs, st.session_state.messages)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": relevant_docs
            })
            
            st.markdown(f'<div class="chat-message assistant-message"><strong>Zama Assistant:</strong><br>{response}</div>', unsafe_allow_html=True)
            
            if relevant_docs:
                with st.expander("当 Source Documents"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f'<div class="source-docs"><strong>Source {i+1} (Distance: {doc["distance"]:.3f}):</strong><br>{doc["content"][:200]}...</div>', unsafe_allow_html=True)
        
        st.rerun()

if __name__ == "__main__":
    main()
