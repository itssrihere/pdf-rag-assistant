# pdf-rag-assistant

A terminal RAG chatbot: point it at a folder of PDFs, then ask questions and get answers with file and page citations.

## Quick start

Requires Python 3.12+, Docker, an OpenAI API key (embeddings), an Anthropic API key (Claude answers; or set `CHAT_PROVIDER=openai`), and a folder of PDFs.

```bash
git clone https://github.com/itssrihere/pdf-rag-assistant.git && cd pdf-rag-assistant

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # set OPENAI_API_KEY, ANTHROPIC_API_KEY, POSTGRES_PASSWORD (any value), DOCS_DIR
docker compose up db -d  # Postgres + pgvector on localhost:5432
python main.py           # ingest PDFs, then chat
```

The first run embeds every PDF (small OpenAI cost); later runs skip PDFs already stored. Details below.

## How it works

1. **Ingest** (`ingest.py`): each PDF is split into overlapping chunks, embedded with OpenAI, and stored in Postgres/pgvector. Bibliographies are dropped first (see below). Chunk IDs are `<collection>:<sha256 of file>:<index>`, so PDFs already stored are skipped (no duplicates, no repeat embedding cost), and the same PDF can live in several collections. Renaming a file doesn't trigger re-ingestion.
2. **Query** (`query.py`): the top `RETRIEVAL_K` chunks are retrieved and passed to the chat model (Claude Haiku 4.5 by default; OpenAI selectable), which answers only from them. Sources print as `file.pdf (pp. 3, 7)` (physical PDF pages, which can differ from printed page numbers).
3. **Memory**: the last `HISTORY_TURNS` Q&A pairs are remembered. Follow-ups are rewritten into standalone questions before retrieval (one extra LLM call per follow-up). Single-question mode (`-q`) has no history.

`main.py` runs ingest, then the interactive chat. Type `exit` or `quit` to stop.

**Why bibliographies are dropped.** In a test over 21 papers, about 22% of chunks were reference-list entries. They match topic queries well (they list titles on the same subject), crowd out real content, and let the model cite a paper's *references* as if they were findings. Ingest cuts each paper at its "References" heading (or numbered list at the end), then drops only later pages that are densely bibliographic, so figure legends and methods that follow are kept. Use `--keep-references` to disable.

```
pdf-rag-assistant/
├── main.py            # Entry point: ingest → interactive Q&A
├── config.py          # Settings from environment / .env
├── ingest.py          # Load, chunk, embed, store (also runnable standalone)
├── query.py           # Retrieval, memory, chat loop (also runnable standalone)
├── llm.py             # Chat-model factory: Claude (default) or OpenAI
├── pdf_text.py        # PDF text extraction (pypdf default; optional pypdfium2)
├── hybrid.py          # BM25 + vector retrieval experiment (not used by the chatbot)
├── eval/              # Retrieval evaluation: questions, metrics, runner (see eval/README.md)
├── tests/             # Unit tests (no network or database needed)
├── requirements.txt   # Runtime dependencies (requirements-dev.txt adds test/experiment extras)
├── Dockerfile         # python:3.12-slim, runs main.py
├── docker-compose.yml # pgvector/Postgres 16 + the app
├── .env.example       # Template for .env (git-ignored)
└── LICENSE            # MIT
```

## Setup details (local)

The Quick start covers the steps. Notes:

- `POSTGRES_PASSWORD` can be any value on first setup; it is applied when the database volume is created.
- `python main.py --docs-dir /path/to/pdfs` overrides `DOCS_DIR`.

Always run inside the venv; system Python won't have the dependencies. Any `DB_URL` already exported in your shell overrides `.env` (`unset DB_URL` if stale).

Other options:

```bash
python main.py --docs-dir /path/to/pdfs --reset   # wipe the collection and re-ingest
python main.py --keep-references                  # also embed bibliographies (dropped by default)
```

Example session:

```
  Skipping (already ingested): old.pdf
  Loading: report.pdf
  Stored 187 chunks from 42 pages
Done. 1 PDF(s) ingested, 1 skipped.

Ask: What is the main topic?

Answer:
The document covers ...

Sources: report.pdf (pp. 3, 7)

Ask: And how was it evaluated?      # follow-up uses chat history
```

## Docker Compose (app + database)

```bash
docker compose up --build         # DOCS_DIR (from .env) is mounted at /docs
```

Compose fails fast if `POSTGRES_PASSWORD` or `OPENAI_API_KEY` is unset; `ANTHROPIC_API_KEY` is needed too unless `CHAT_PROVIDER=openai`.

## Running steps separately

```bash
python ingest.py --docs-dir /path/to/pdfs [--reset]   # ingest only
python query.py                                       # chat against existing data
python query.py -q "What is the main topic?"          # one question, then exit
```

## Configuration

Read from environment variables, or from `.env` (real environment variables take precedence).

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key; always used for embeddings (Anthropic has no embedding model), and for chat if `CHAT_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | *(required unless `CHAT_PROVIDER=openai`)* | Anthropic API key for chat |
| `CHAT_PROVIDER` / `CHAT_MODEL` | `anthropic` / `claude-haiku-4-5` | `openai` selects `gpt-4o-mini`; `CHAT_MODEL` overrides the model. Newer Claude models (Sonnet/Opus 5.x) reject a custom temperature, so it is only sent to models that accept it |
| `POSTGRES_PASSWORD` | *(required unless `DB_URL` set)* | Postgres password |
| `POSTGRES_USER` / `POSTGRES_DB` | `postgres` / `rag` | Postgres user and database |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `localhost` / `5432` | Compose sets host to `db` for the app container |
| `DB_URL` | built from `POSTGRES_*` | Full connection string; overrides the above |
| `DOCS_DIR` | *(required if `--docs-dir` not passed)* | Folder scanned recursively for `*.pdf` (any case); hidden folders and `._*` files are skipped. Sources are cited by path relative to this folder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Characters per chunk / overlap |
| `RETRIEVAL_K` | `8` | Chunks retrieved per query (8 beat 4 in the retrieval eval; see `eval/`) |
| `COLLECTION_NAME` | `pdf_docs` | pgvector collection name; use different names to keep variants side by side |
| `HISTORY_TURNS` | `5` | Past Q&A pairs remembered in a chat |
| `MAX_DOCS` | `500` | Most PDFs one ingest run processes, in path order (`--max-docs`; `0` = no limit). A guard against embedding a huge folder by accident; anything beyond it is reported, not silently dropped |
| `PDF_EXTRACTOR` | `pypdf` | `pypdfium2` is an optional alternative (`pip install pypdfium2`); see Tests and evaluation for why `pypdf` stays the default |

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
pytest                                   # 112 unit tests; no database or API key needed
python -m eval.run_eval --ks 4 8 12 16   # retrieval evaluation (needs the database and OPENAI_API_KEY)
```

`eval/` holds 42 questions over the example corpus (specific facts, "which study" disambiguation, multi-paper summaries, exact-term lookups, out-of-corpus). Retrieval was measured on 21 papers; "right page" means the page containing the answer was among the top *k* chunks.

| Measure (k = chunks retrieved) | With references, k=4 (original) | With references, k=8 | References removed, k=8 |
|---|---|---|---|
| Right paper retrieved (all question types) | 100% | 100% | 100% |
| Right page, specific-fact questions | 88% | 94% | 94% |
| Multi-paper questions: share of relevant papers found | 83% | 83% | 94% |
| Autophagy summary: share of the 6 relevant papers found | 50% | 50% | 83% |

Findings: raising `k` from 4 to 8 improved page-level retrieval; removing bibliographies fixed most of the multi-paper gap and made answers cite the papers themselves instead of entries from their reference lists. MMR (diversified retrieval, measured before bibliographies were removed) helped summaries at k=8 and above but lowered page accuracy on fact questions by 32 points at k=4 and 25 at k=8, so it is not the default (`--strategy mmr` in the eval harness). **A second, non-paper corpus.** To check the app is not tied to research papers, the Spring Boot 3.1.12 reference (939 pages, split into 112 PDFs across nested folders) was ingested with the same default settings: nothing was retuned, no pages were mistaken for bibliographies, and the table of contents made up only 3% of retrieved chunks. Retrieval of the right section (`python -m eval.run_eval --questions eval/spring_questions.jsonl`, 42 questions, vector search):

| Question type | k=4 | k=8 |
|---|---|---|
| Facts, property names and "which section" questions, using the docs' own wording (26) | 100% | 100% |
| Everyday phrasing with none of the docs' vocabulary (8) | 62% | 75% |
| Multi-section topics, share of relevant sections found (4) | 50% | 75% |
| Out-of-scope questions declined (4) | 4/4 | |

Keyword (BM25) search was worse on the everyday-phrasing questions (38% / 50%), and hybrid search only matched vector search (62% / 62%), so the earlier decision to stay vector-only held on this corpus too. The misses were vocabulary gaps ("requests dropped on redeploy" versus the docs' "graceful shutdown"), and in two cases the retriever returned another section that also answers the question, which a single-passage ground truth counts as a miss. The question set is small and written by the same author who built the corpus, so treat these as indications, not benchmarks.

**Alternative PDF text extractors were tested and not adopted.** pypdfium2 and PyMuPDF removed all 137 glyph-corruption artifacts (e.g. `/H11032` for a prime mark) and cut fused words by 20-30% across the 488 pages, but retrieval did not improve: the same questions were missed, and differences (an exact-term question at k=4, one paper in a multi-paper answer) were within noise. Multi-paper recall at k=8 was slightly lower (89% vs 94%), probably because shifted chunk boundaries change which chunks rank highest. pypdf stays the default; `PDF_EXTRACTOR=pypdfium2` remains as an option. (PyMuPDF is AGPL-licensed and was dropped; pdfplumber lost word spacing and was ruled out.) The exact-term questions exclude fused-token cases, so they cannot see the symbol-corruption benefit.

**Chunk size and a paper-title prefix were tested and not adopted.** Re-ingesting at 300 and 1000 characters (about 10% overlap) gave no clear gain on single-fact questions (page accuracy at k=8 was 94% for all three sizes). Larger chunks hurt multi-paper coverage, since fewer distinct chunks means fewer distinct papers per k (recall at k=8: 94% at 500 characters, 72% at 1000); at roughly equal context size, 300 and 500 characters were about even. Embedding each chunk as "Paper: <title> + text" (while storing and showing only the original text) was clearly worse: multi-paper recall fell to 44% at every k and page accuracy at k=4 from 88% to 69%, because the shared title made every chunk of the best-matching paper outrank other papers' chunks (for two of the three multi-paper questions, all 12 retrieved chunks came from one paper). The defaults (500 characters, 50 overlap, file and page metadata on every chunk) stay.

**Hybrid search (keyword + vector) was tested and not adopted.** `hybrid.py` fuses BM25 keyword ranking with vector search by reciprocal rank fusion (`--strategy hybrid`; `bm25` = keyword only). Keyword-only search was best on 13 exact-term questions (gene names, SNP IDs, catalog numbers: right paper in the top 4 for 92% vs 77% for vector search) but worst on multi-paper summaries (recall at k=8: 83% vs 94%). Rank fusion landed in between: +1 question on page-level accuracy for fact questions, no gain on exact terms at k=8, and the same loss on summaries. Weighting the keyword list 2-3x helped at k=4 (exact terms 77% to 85%) but not at k=8, because rank fusion rewards chunks both lists agree on, so one decisive exact-match hit can still lose to generic overlap. A larger `k` (12) gave vector search 92% on exact terms with no summary penalty, so the chatbot stays vector-only.

**Answer-level checks** (`--judge`, 38 answerable questions, answers at k=8, `gpt-4o-mini` as judge). Faithfulness (share of the answer's claims supported by the retrieved text) and answer relevancy (1-5) were at the ceiling: 0.98-1.00 and 4.9-5.0 on both collections, so they don't discriminate much here. A negative control confirmed the judge isn't a rubber stamp: appending two fabricated claims to real answers dropped faithfulness from 1.00 to 0.33-0.50, and an off-topic answer scored 1/5. What *did* discriminate is a deterministic citation check (do cited papers and pages match what was retrieved?):

| Multi-paper answers (3 questions) | With references in the index | References removed |
|---|---|---|
| Citations grounded in a retrieved paper | 6 of 17 | 13 of 13 |
| Author-year citations to outside works | 11 | 0 |

Faithfulness scored both versions 1.00, because the outside citations were faithful to the retrieved text (it was a reference-list title); only the citation check caught the misattribution. Context precision (share of the top-k chunks from the right paper) is 94% at k=4 and 89% at k=8 for fact questions, and about 52-58% for exact-term questions. Only about half of the answers cite inline at all (the chat prints a Sources line regardless), and caveats apply: the judge is the same model family as the generator (self-preference bias), and it grades against the retrieved context only.

With only 42 questions, treat differences of a few points as noise; see `eval/README.md` for scoring rules and caveats.

## Purpose

A learning project: I built it to get hands-on with Python and its ecosystem (LangChain, Postgres/pgvector, pytest, Docker) by going past a tutorial-style demo: measuring retrieval quality, testing alternatives, and keeping the results that did not pan out in the write-up.

## AI assistance

This project was built with Claude Code (Anthropic). I set the goals, made the design decisions, and judged the results; Claude Code wrote most of the code and tests under that direction. Every number in this README comes from runs of the included evaluation, and the experiments that did not help are reported alongside the ones that did.

## Notes and limitations

- **One bad PDF doesn't stop ingestion.** A file that fails (corrupt, unreadable, or an OpenAI connection, timeout or rate-limit error while embedding) is logged to stderr with its root cause and skipped, and a summary at the end lists every failure, every PDF with no extractable text (scans; there is no OCR), and anything left out by `MAX_DOCS`. `python ingest.py` exits with status 1 if any file failed. Five failures in a row stop the run early, since that usually means the database or API key is the problem (with the client's retries that takes a couple of minutes during a total outage). A failed file leaves nothing partial behind, so re-running retries only what is missing. In the chat, a question that fails (for example the API is unreachable) prints an error and the session continues.

- **Password is set once.** Postgres applies `POSTGRES_PASSWORD` only when the data volume is created. To change it later, run `ALTER USER postgres PASSWORD '...'` (e.g. via `docker exec <db-container> psql -U postgres`), or `docker compose down -v` (deletes all ingested data).
- **Edited PDFs:** a changed file gets a new hash and is ingested as new chunks; the old version's chunks remain until you run with `--reset`.
- **Behind a corporate proxy that inspects TLS?** If you see `CERTIFICATE_VERIFY_FAILED`, point `SSL_CERT_FILE` at your organization's CA bundle.
- **Changing chunk settings or the reference-stripping option** doesn't re-process already-ingested files (they're skipped by content hash); use `--reset`, or ingest into a new `COLLECTION_NAME`.
- **Reference detection is heuristic.** It handles "References", "References and notes", "Bibliography" and heading-less numbered lists at the end of a paper; unusual layouts may keep some bibliography or drop a page. Check `Dropped N reference page(s)` in the ingest output.
