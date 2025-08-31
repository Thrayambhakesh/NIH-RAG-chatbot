import streamlit as st
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import ChatTogether  # ✅ Together.ai support
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Load Vector DB using the same embedding model used in chunking
vectorstore = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="./local_miniLM"),
    persist_directory="./chroma.db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)

# 2. Load LLM from Together API using Mistral 7B
llm = ChatTogether(model="mistralai/Mistral-7B-Instruct-v0.2")

retriever = vectorstore.as_retriever()

# 3. Prompt template
system_prompt = (
    "You are a friendly clinician at the National Institutes of Health (NIH). "
    "Your task is to answer clients' questions as truthfully as possible. "
    "Use the retrieved information from the NIH website to help answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep your answer under 3 sentences.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# 4. Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Streamlit UI
st.set_page_config(page_title="NIH Chatbot", page_icon="🧠")
st.title("🩺 NIH RAG Chatbot - Powered by Llama 3 and HuggingFace")

st.info(
    "This chatbot uses the NIH ODS Fact Sheets (https://ods.od.nih.gov/api/) "
    "to generate accurate medical responses using Llama 3 from Together.ai."
)

query = st.chat_input(placeholder="Your search query...")

if query:
    with st.chat_message("user"):
        st.write(query)

    result = rag_chain.invoke({"input": query})

    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.write("**Sources:**")
        for doc in result["context"]:
            st.write(f'- {doc.metadata["source"].replace(" ", "%20")}')
#$env:TOGETHER_API_KEY="2bf53a903c4617865524e9d48c9dbe785f34349c61e260c8191e413936d52202"
