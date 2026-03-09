import streamlit as st
from app import load_and_split_pdf, build_vectorstore, create_rag_agent, chat

st.set_page_config(page_title="PDF Chat Agent", page_icon="📄")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing PDF..."):

        chunks = load_and_split_pdf("temp.pdf")
        vectorstore = build_vectorstore(chunks)
        agent = create_rag_agent(vectorstore)

    st.success("PDF Ready!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask something about the PDF"):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat(agent, prompt)

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
