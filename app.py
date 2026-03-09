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
    You are a helpful assistant that answers questions using the provided PDF knowledge base.

    IMPORTANT INSTRUCTIONS:
    1. Always use the retrieval tool to search the PDF first
    2. Read and understand the retrieved context carefully
    3. Provide CLEAR, CONCISE, and DIRECT answers
    4. Answer only what is asked - no extra information
    5. Use simple language and short sentences
    6. If the answer is a list, use bullet points
    7. Do NOT return raw chunks or metadata from the PDF
    8. Synthesize information into a natural, human-readable response

    Example:
    Question: "In which language does the person work?"
    Bad: [Returns full resume chunks with metadata]
    Good: "The person works with JavaScript, TypeScript, and SQL."
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
    """
    Execute agent and return only the final AI response,
    filtering out all tool calls and intermediate steps.
    """
    
    messages = []
    
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        messages = event["messages"]
    
    # Get the last AI message (skip tool messages)
    for message in reversed(messages):
        if hasattr(message, "type") and message.type == "ai":
            if hasattr(message, "content"):
                if isinstance(message.content, str) and message.content:
                    return message.content
                elif isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part["text"]
    
    return "I couldn't generate a response."
