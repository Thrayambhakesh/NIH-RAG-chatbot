# STEP 1: NIH HTML scraping
import requests
from bs4 import BeautifulSoup

NIH_API = "https://ods.od.nih.gov/api/" 

response = requests.get(NIH_API)
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find_all("a", string="HTML")

# Filter out Spanish and make full paths
web_paths = [NIH_API + link["href"] for link in links if "espanol" not in link["href"]]

# STEP 2: Load web content
from langchain_community.document_loaders import WebBaseLoader

print("Fetching content...")
loader = WebBaseLoader(
    web_paths=web_paths,
    requests_kwargs={
        "headers": {
            "User-Agent": "my-nih-rag-bot/0.1 (contact: 127003287@sastra.in)"
        }
    }
)
docs = loader.load()

# STEP 3: Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# STEP 4: Save to vector DB using HuggingFace embeddings (NOT OpenAI)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings

embedding = HuggingFaceEmbeddings(model_name="./local_miniLM")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory="./chroma.db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )
)
