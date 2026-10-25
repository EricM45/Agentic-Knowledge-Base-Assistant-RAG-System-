"""Browser chat for the document RAG pipeline. Run: streamlit run app.py"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ragbot.config import Settings
from ragbot.gold import SUGGESTIONS
from ragbot.loader import SUPPORTED, collect_files
from ragbot.pipeline import RagPipeline
from ragbot.store import corpus_signature

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _name in ("httpx", "httpx2", "httpcore", "httpcore2", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

CSS = """
<style>
.block-container { max-width: 760px; padding-top: 1.5rem; }
[data-testid="stSidebar"] { border-right: 1px solid #e4ddd2; }
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Document RAG Chatbot", page_icon="💬", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Document RAG Chatbot")
    st.caption("Answers are grounded in your files. Each reply cites the passage it came from.")

    settings = Settings.from_env()
    mode, top_k, backend, openai_key, ollama_model, pinecone_key, pinecone_index = _sidebar(settings)
    paths = _selected_paths(settings, mode)
    if not paths:
        st.info("Upload a PDF, Word, Markdown, or text file to start.")
        return

    files = collect_files(paths)
    if not files:
        st.info("None of those files are PDF, DOCX, Markdown, or text.")
        return

    signature = corpus_signature(files, settings, settings.embedding_model)
    try:
        with st.spinner(
            "Indexing documents. The first launch downloads the embedding model (about 90 MB) and a reranker (about 80 MB)."
        ):
            pipeline = _load_pipeline(signature, tuple(str(path) for path in files), settings.embedding_model)
    except Exception as exc:
        st.error("The embedding model could not be loaded. Check the network connection and run the app again.")
        st.caption(f"{exc.__class__.__name__}: {exc}")
        return

    pipeline.configure(
        openai_api_key=openai_key or None,
        ollama_model=ollama_model or None,
        backend=backend,
        pinecone_api_key=pinecone_key or None,
        pinecone_index=pinecone_index or "rag-chatbot",
    )
    if pipeline.needs_pinecone_sync():
        with st.spinner("Saving passage vectors in Pinecone…"):
            pipeline.ensure_pinecone()
    else:
        pipeline.ensure_pinecone()
    if pipeline.vector_store_error:
        st.warning(pipeline.vector_store_error)
    if st.session_state.get("active_signature") != signature:
        st.session_state.active_signature = signature
        st.session_state.messages = []

    st.caption(
        f"{pipeline.stats.files} files · {pipeline.stats.chunks} passages indexed · {pipeline.backend_label()} · {pipeline.vector_store_label()}"
    )
    if mode != "My files only":
        st.caption("Sample knowledge base: Helios Labs, a fictional company.")
    for note in pipeline.stats.skipped:
        st.warning(note)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if not st.session_state.messages and mode != "My files only":
        _suggestion_buttons()

    override = st.session_state.pop("queued_question", None)
    for message in st.session_state.messages:
        _render_message(message)

    typed = st.chat_input("Ask a question about the documents")
    question = override or typed
    if not question:
        return

    history = [{"role": item["role"], "content": item["content"]} for item in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching the documents…"):
        answer = pipeline.ask(question, history=history, top_k=top_k)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "backend": answer.backend,
            "backend_label": f"{pipeline.backend_label()} · {pipeline.vector_store_label()}",
            "refused": answer.refused,
            "warning": answer.warning,
            "question": question,
            "standalone": answer.standalone_question,
            "query_terms": answer.query_terms,
            "expanded_terms": answer.expanded_terms,
            "vector_store": answer.vector_store,
            "sources": answer.sources,
        }
    )
    st.rerun()


def _sidebar(settings: Settings):
    with st.sidebar:
        st.header("Knowledge base")
        mode = st.radio(
            "Documents",
            ["Sample knowledge base", "Sample plus my files", "My files only"],
            label_visibility="collapsed",
        )
        uploaded = st.file_uploader(
            "Add files",
            type=["pdf", "docx", "md", "markdown", "txt"],
            accept_multiple_files=True,
            help="PDF, Word, Markdown, or plain text.",
        )
        if uploaded:
            st.session_state.upload_paths = [str(path) for path in _save_uploads(uploaded)]
        elif uploaded is not None:
            st.session_state.upload_paths = []
        top_k = st.slider("Passages to retrieve", min_value=2, max_value=8, value=settings.top_k)
        st.divider()
        st.header("Answer engine")
        backend_label = st.selectbox("Backend", ["Auto", "Local NLP", "OpenAI", "Ollama"])
        for secret_key in ("openai_key", "openai_key_override", "pinecone_key", "pinecone_key_override", "pinecone_index_name"):
            st.session_state.pop(secret_key, None)
        if "ollama_model" not in st.session_state:
            st.session_state.ollama_model = settings.ollama_model or ""
        ollama_model = st.text_input("Ollama model", key="ollama_model", placeholder="gemma4:31b-cloud")
        if backend_label == "Auto":
            st.caption("Uses a configured model when one is available.")
        st.divider()
        st.header("Vector database")
        st.caption("Pinecone" if settings.pinecone_api_key else "Local vectors")
        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()
    backend = {"Auto": "auto", "Local NLP": "local-nlp", "OpenAI": "openai", "Ollama": "ollama"}[backend_label]
    return (
        mode,
        top_k,
        backend,
        settings.openai_api_key or "",
        ollama_model.strip(),
        settings.pinecone_api_key or "",
        settings.pinecone_index,
    )


def _selected_paths(settings: Settings, mode: str) -> list[Path]:
    uploads = [Path(path) for path in st.session_state.get("upload_paths", [])]
    sample = [settings.sample_dir]
    if mode == "My files only":
        return uploads
    if mode == "Sample plus my files":
        return sample + uploads
    return sample


def _save_uploads(files) -> list[Path]:
    destination = ROOT / "data" / "uploads"
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for item in files:
        name = Path(item.name).name
        if Path(name).suffix.lower() not in SUPPORTED:
            continue
        target = destination / name
        payload = item.getvalue()
        if not target.exists() or target.read_bytes() != payload:
            target.write_bytes(payload)
        paths.append(target)
    return paths


def _suggestion_buttons() -> None:
    st.caption("Try a question from the sample knowledge base.")
    for suggestion in SUGGESTIONS:
        st.button(suggestion, on_click=_queue, args=(suggestion,), width="stretch")


def _queue(text: str) -> None:
    st.session_state.queued_question = text


def _render_trace(message: dict) -> None:
    """Show the retrieval, generation, context, and NLP steps behind one reply."""
    with st.expander("How this answer was produced"):
        store = message.get("vector_store") or "local"
        retrieval = "Pinecone nearest-neighbor search" if store == "pinecone" else "Local vector search"
        st.markdown(f"**Retrieval.** {retrieval}, fused with BM25 keyword ranks, then reranked.")
        backend = message.get("backend") or ""
        if backend in {"ollama", "openai"}:
            st.markdown("**Generation.** A language model wrote this reply from the retrieved passages only.")
        elif backend == "local-nlp":
            st.markdown("**Generation.** The reply was composed by selecting the retrieved sentences that best match the question.")
        else:
            st.markdown("**Generation.** No answer was generated because the documents did not support one.")
        standalone = message.get("standalone") or ""
        question = message.get("question") or ""
        if standalone and standalone != question:
            st.markdown(f"**Context.** The follow-up was searched as: {standalone}")
        else:
            st.markdown("**Context.** The question was searched as written. Earlier turns are still sent to the language model.")
        terms = message.get("query_terms") or ()
        if terms:
            line = "**Applied NLP.** Stemmed terms: " + ", ".join(terms)
            extras = message.get("expanded_terms") or ()
            if extras:
                line += ". Synonyms added for search: " + ", ".join(extras)
            st.markdown(line + ".")


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("warning"):
            st.caption(message["warning"])
        label = message.get("backend_label") or message.get("backend", "")
        if label:
            st.caption(label)
        _render_trace(message)
        sources = message.get("sources") or ()
        if not sources:
            return
        title = "Nearest passages" if message.get("refused") else "Sources"
        with st.expander(title):
            standalone = message.get("standalone")
            if standalone and standalone != message.get("question"):
                st.caption(f"Search query: {standalone}")
            for source in sources:
                location = source.source
                if source.section:
                    location += f" · {source.section}"
                if source.page:
                    location += f" · p. {source.page}"
                st.markdown(f"**[{source.number}] {location}**")
                scores = [f"dense {source.dense_score:.2f}", f"BM25 {source.sparse_score:.2f}"]
                if source.rerank_score is not None:
                    scores.append(f"rerank {source.rerank_score:.2f}")
                st.caption(" · ".join(scores))
                snippet = source.text.strip()
                if len(snippet) > 500:
                    snippet = snippet[:500].rsplit(" ", 1)[0].rstrip() + "…"
                st.write(snippet)


@st.cache_resource(show_spinner=False)
def _load_pipeline(signature: str, files: tuple[str, ...], model_name: str) -> RagPipeline:
    del signature, model_name
    settings = Settings.from_env()
    pipeline = RagPipeline(settings)
    path_list = [Path(file) for file in files]
    if not pipeline.load_cached(path_list):
        pipeline.index_paths(path_list)
    else:
        pipeline.ensure_pinecone()
    pipeline.warm()
    return pipeline


main()
