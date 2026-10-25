# Document RAG Chatbot

A Python chatbot that answers questions from your own files. It retrieves the relevant passages, then writes an answer that cites them.

The sample knowledge base is **Helios Labs**, a fictional company. Add your own PDF, Word, Markdown, or text files from the sidebar.

## What this project demonstrates

**Retrieval-augmented generation.** Each reply is written from passages retrieved for that question, and the reply cites those passages. OpenAI or Ollama generates the wording when configured. The source expander shows the passages and their dense, BM25, and reranker scores.

**Context-aware responses.** A follow-up that uses "it", "they", or "what about" is rewritten with the previous question before search. The language model also receives the recent conversation, so the reply can stay on the same topic.

**Applied NLP.** Questions are tokenized, stop words are removed, and a small stemmer plus synonym list (for example "PTO" and "vacation" to "paid time") improves keyword search. A passage is used only when those terms, or the embedding and reranker scores, support it.

Open **How this answer was produced** under any reply to see those four steps for that question.

## Run it

The project uses Python 3.11. From this folder:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Open the local URL Streamlit prints, usually http://localhost:8501.

In VS Code or Cursor:

1. Open this folder.
2. Select the interpreter `.venv/bin/python` if the editor asks. That path is already set in `.vscode/settings.json`.
3. Open **Run and Debug** and start **RAG Chatbot (browser UI)**, or press F5.

**RAG Chatbot (terminal)** starts a chat in the integrated terminal. **Evaluate sample questions** checks the Helios questions end to end. **Run chatbot** in the Terminal menu starts the browser UI.

```bash
python cli.py
python cli.py --question "What is the standard warranty on the Helios Arc inverter?"
python -m unittest discover -s tests -v
python -m ragbot.evaluate
.venv/bin/pyrefly check
```

The first launch downloads two small models: `all-MiniLM-L6-v2` (about 90 MB) and a cross-encoder reranker (about 80 MB).

## Secrets

Copy `.env.example` to `.env` and fill in keys there. The app reads that file and does not display the values. `.env` is listed in `.gitignore`, so it stays off GitHub. Publish `.env.example` only.

## What happens to a question

1. A follow-up such as "How long can it be extended?" is rewritten with the previous user question so retrieval still sees the topic.
2. Documents are split by heading, then into overlapping sentence chunks.
3. Each chunk is embedded with a sentence-transformer model. With `PINECONE_API_KEY` set, those vectors are stored in Pinecone and nearest-neighbor search runs there. Keyword search (BM25) stays on this computer.
4. Dense ranks and sparse ranks are fused with reciprocal rank fusion. A cross-encoder reranks that pool when `USE_RERANKER` is on.
5. A passage is used when rare question terms (or a small set of English expansions such as "PTO" for "paid time") appear in it, or when the reranker and embedding scores both clear their thresholds.
6. Ollama or OpenAI writes the answer when one is configured. Otherwise the app selects the most relevant sentences and cites them as `[1]`, `[2]`.

### Environment

| Variable | Purpose |
| --- | --- |
| `OLLAMA_MODEL` | Model name. Use `gemma4:31b-cloud` after `ollama signin`, or `llama3.2` for a model installed on this computer. |
| `OLLAMA_BASE_URL` | Local Ollama server. Default `http://127.0.0.1:11434/v1`. |
| `OPENAI_API_KEY` | Optional hosted OpenAI answers. |
| `OPENAI_MODEL` | Default `gpt-4o-mini`. |
| `PINECONE_API_KEY` | Optional. Enables the Pinecone vector index. |
| `PINECONE_INDEX` | Index name. Default `rag-chatbot`. |
| `PINECONE_CLOUD` | Default `aws`. |
| `PINECONE_REGION` | Default `us-east-1`. |
| `EMBEDDING_MODEL` | Default `all-MiniLM-L6-v2`. |
| `CHUNK_SIZE` | Characters per passage. Default `800`. |
| `CHUNK_OVERLAP` | Overlap between passages. Default `140`. |
| `TOP_K` | Passages sent to the answer step. Default `4`. |
| `USE_RERANKER` | Set to `0` to skip the cross-encoder. |

Cloud Ollama models need a signed-in Ollama app. Run `ollama signin` and approve the device in the browser. The account password is not stored in this project.

Without a Pinecone key, vectors stay in `data/index/` on this computer.

## Sample questions

- What is the standard warranty on the Helios Arc inverter?
- How many PTO days does a new employee receive?
- What should I do if an Arc inverter shows error E17?
- What is the monthly price of Gridline monitoring?
- Who is the chief executive of Helios Labs?

After the warranty question, ask "How long can it be extended?" The second answer should mention 25 years.

## Checks

Pyrefly (`meta.pyrefly`) is the editor type checker. It is recommended in `.vscode/extensions.json` and configured in `pyproject.toml`. Run `.venv/bin/pyrefly check` before publishing.

`python -m unittest discover -s tests -v` covers chunking, retrieval, grounding, and the Pinecone upload path with a stand-in vector store. `python -m ragbot.evaluate` runs the sample questions with the real embedding model.

## Project layout

- `app.py` — Streamlit chat
- `cli.py` — terminal chat
- `ragbot/` — loader, chunker, BM25, embeddings, Pinecone, fusion, grounding, generation
- `sample_documents/` — fictional Helios Labs corpus
- `tests/test_rag.py` — tests that do not download models
- `pyproject.toml` — Pyrefly configuration
- `.vscode/` — interpreter, debugger, tasks, and the Pyrefly recommendation

Uploaded files are copied to `data/uploads/`. The local index cache lives in `data/index/`. Both directories are gitignored except for their placeholder files.
