import utils
import streamlit as st
import os
import openai
# openai.api_key = utils.get_openai_api_key()
openai.api_key = st.secrets.OPENAI_API_KEY

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./SIGMAN_Camouflage_SOP.pdf"]
).load_data()

from llama_index import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

import os
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index import load_index_from_storage


def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=40,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

from llama_index.llms import OpenAI

index = build_sentence_window_index(
    [document],
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),
    save_dir="./sentence_index",
)



import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# openai.api_key = st.secrets.openai_key

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1,system_prompt="Keep your answers technical and based on facts – do not hallucinate features.",api_key=openai.api_key)

st.title("Chat with the Streamlit docs, powered by LlamaIndex 💬🦙")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question "}
    ]

@st.cache_resource(show_spinner=False)

def load_data():
    try:
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            documents = SimpleDirectoryReader(input_files=["./SIGMAN_Camouflage_SOP.pdf"]).load_data()
            
            llm = OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt="Keep your answers technical and based on facts – do not hallucinate features.", api_key=openai.api_key)
            
            index = build_sentence_window_index([document],llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),save_dir="./sentence_index",)
            
            
            
            return index
    except Exception as e:
        st.error(f"Error loading and indexing data: {e}")
        return None

automerging_index = load_data()

if automerging_index:
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = automerging_index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                user_input = st.session_state.messages[-1]["content"]
                
                # Query automerging_query_engine
                query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
                window_res = query_engine.query(user_input)
                
                st.write(str(window_res))

            

                # Add both responses to the message history
                message_auto_merging = {"role": "assistant", "content": str(window_res)}
                
                st.session_state.messages.append(message_auto_merging)
               
