import streamlit as st
from Doc_processor.file_handler import DocProcessor
from retriever.vectordb import HybridRetrieverBuilder
from agents.workflow import QAPipeline
from utils.logging import logger

st.set_page_config(page_title="GyanSetu", layout="centered", initial_sidebar_state="expanded")

# --- Session state defaults ---
for key, val in [("retriever", None), ("pipeline", None), ("chat_history", []), ("chunk_count", 0)]:
    st.session_state.setdefault(key, val)

# --- Sidebar ---
with st.sidebar:
    st.title("GyanSetu")
    st.caption("Powered by Docling, LangGraph, Llama, ChromaDB")
    st.divider()

    if st.session_state.retriever:
        st.success(f"Ready — {st.session_state.chunk_count} chunks indexed")
    else:
        st.info("Waiting for documents")

    st.divider()
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        if st.button("Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    chunks = DocProcessor().process(uploaded_files)
                    st.session_state.retriever = HybridRetrieverBuilder().build(chunks)
                    st.session_state.pipeline = QAPipeline()
                    st.session_state.chunk_count = len(chunks)
                    st.session_state.chat_history = []
                    st.success(f"Done. {len(chunks)} chunks ready.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()

    if st.session_state.retriever:
        if st.button("Reset", use_container_width=True):
            for key in ("retriever", "pipeline"):
                st.session_state[key] = None
            st.session_state.chat_history = []
            st.session_state.chunk_count = 0
            st.rerun()

# --- Main area ---
if not st.session_state.retriever:
    st.title("GyanSetu")
    st.write("Welcome. Upload documents from the sidebar to get started.")
    st.divider()
    for label, title, delta in [("Step 1", "Upload", "Add documents"),
                                  ("Step 2", "Process", "Chunk & index"),
                                  ("Step 3", "Ask", "Query your data")]:
        st.metric(label, title, delta)  # use columns if layout matters

else:
    st.title("Chat")
    st.caption(f"{st.session_state.chunk_count} chunks indexed. Ask a question about your documents.")
    st.divider()

    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            with st.expander("Verification Report"):
                st.markdown(entry["report"])

    if question := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not st.session_state.pipeline:
                        st.session_state.pipeline = QAPipeline()

                    result = st.session_state.pipeline.run(question, st.session_state.retriever)
                    answer, report = result["draft_answer"], result["verification_report"]

                    st.write(answer)
                    with st.expander("Verification Report"):
                        st.markdown(report)

                    st.session_state.chat_history.append(
                        {"question": question, "answer": answer, "report": report}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"Pipeline error: {e}")