import os
from dotenv import load_dotenv
import bs4

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent


# ==========================
# Load environment variables
# ==========================

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found")


# ==========================
# Initialize Gemini model
# ==========================

model = init_chat_model("google_genai:gemini-2.5-flash-lite")


# ==========================
# Embedding model
# ==========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ==========================
# Vector Store
# ==========================

vector_store = InMemoryVectorStore(embeddings)


# ==========================
# Load Documents
# ==========================

bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

print(f"Loaded {len(docs)} document")


# ==========================
# Split Documents
# ==========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splits = text_splitter.split_documents(docs)

print(f"Split into {len(splits)} chunks")


# ==========================
# Add to Vector Store
# ==========================

vector_store.add_documents(splits)

print("Documents indexed")


# ==========================
# Retrieval Tool
# ==========================

@tool
def retrieve_context(query: str):
    """Retrieve information from the knowledge base."""

    docs = vector_store.similarity_search(query, k=2)

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in docs
    )

    return serialized


# ==========================
# Create Agent
# ==========================

tools = [retrieve_context]

system_prompt = (
    "You are a helpful assistant with access to a tool "
    "that retrieves context from a knowledge base."
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt
)


# ==========================
# Run Agent
# ==========================

print("\n===== RAG Agent Ready =====")

while True:

    query = input("\nAsk a question: ")

    if query.lower() in ["exit", "quit"]:
        break

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
