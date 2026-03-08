import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
MODEL_NAME     = "gemini-2.5-flash"   

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent PDF assistant. Use the context below to answer the question.
If the answer isn't in the context, say "I couldn't find that in the document."
Always be concise, accurate, and helpful.

Context:
{context}

Question: {question}

Answer:""",
)


def load_and_split_pdf(pdf_path: str):
    """Load PDF and split into chunks."""
    print(f"\n Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f" Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f" Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and store in FAISS vector DB."""
    print("\n Building vector store with Gemini Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(" Vector store ready")
    return vectorstore


def create_rag_chain(vectorstore):
    """Create conversational RAG chain with Gemini LLM."""
    print("\n Initialising Gemini RAG chain...")
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    print(" Chain ready — let's chat!\n")
    return chain


def chat(chain, question: str) -> str:
    """Send a question to the RAG chain and return the answer."""
    result = chain.invoke({"question": question})
    answer = result["answer"]

    # Show source pages for transparency
    sources = result.get("source_documents", [])
    pages = sorted({doc.metadata.get("page", "?") + 1 for doc in sources})
    source_info = f"Sources: pages {pages}" if pages else ""

    return answer, source_info



def run_chat(pdf_path: str):
    chunks      = load_and_split_pdf(pdf_path)
    vectorstore = build_vectorstore(chunks)
    chain       = create_rag_chain(vectorstore)

    print("=" * 60)
    print("  PDF Chat Agent  —  powered by Gemini + LangChain")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("\n Goodbye!")
            break

        answer, source_info = chat(chain, question)
        print(f"\n Agent: {answer}")
        if source_info:
            print(source_info)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python app.py <path_to_pdf>")
        print("Example: python app.py documents/report.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    run_chat(pdf_path)
