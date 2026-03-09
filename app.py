import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain.tools import tool
from langchain.agents import create_agent

# =========================
# Load ENV
# =========================

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found")


# =========================
# Initialize LLM
# =========================

model = init_chat_model("google_genai:gemini-2.5-flash")


# =========================
# Embedding Model
# =========================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# =========================
# Load PDF
# =========================

def load_and_split_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    return splits


# =========================
# Build Vector Store
# =========================

def build_vectorstore(chunks):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# =========================
# Create Retrieval Tool
# =========================

def create_retrieval_tool(vectorstore):

    @tool
    def retrieve_context(query: str):
        """Search information inside the PDF"""

        docs = vectorstore.similarity_search(
            query,
            k=3
        )

        context = "\n\n".join(
            f"Page: {doc.metadata}\n{doc.page_content}"
            for doc in docs
        )

        return context

    return retrieve_context


# =========================
# Create Agent
# =========================

def create_rag_agent(vectorstore):

    retrieval_tool = create_retrieval_tool(vectorstore)

    tools = [retrieval_tool]

    system_prompt = """
    You are a helpful assistant that answers questions
    using the provided PDF knowledge base.

    Always use the retrieval tool before answering.
    """

    agent = create_agent(
        model,
        tools,
        system_prompt=system_prompt
    )

    return agent


# =========================
# Chat Function
# =========================

def chat(agent, query):

    final_response = ""

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):

        message = event["messages"][-1]

        if hasattr(message, "content"):
            if isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        final_response += part["text"]
            else:
                final_response = message.content

    return final_response
