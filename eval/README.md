# Retrieval evaluation

`python -m eval.run_eval [--ks 4 8 16] [--strategy similarity|mmr|hybrid|bm25] [--bm25-weight 2] [--answers [--show]] [--refusal] [--json out.json]` scores retrieval against
`questions.jsonl` using whichever collection `COLLECTION_NAME` points at. It needs the database and
`OPENAI_API_KEY` (query embeddings only, a few cents; `--refusal` also makes a handful of chat calls).

## Question format (`questions.jsonl`, one JSON object per line)

| Field | Used by | Meaning |
|---|---|---|
| `id`, `category`, `question` | all | `category` is one of the four below |
| `expected_file`, `expected_pages` | specific, disambiguation | the right paper, and every 1-based page where the answer appears |
| `expected_files` | multi_document | papers that substantially cover the topic (50+ mentions of its key terms) |

Expected files and pages were located by searching the PDF text for an exact phrase, not recalled from memory.

## Categories and scoring

- **specific**: one fact in one place. Reported: file hit@k, page hit@k, MRR.
- **disambiguation**: "which study...?" among papers on overlapping topics. Any page of the right paper that
  describes the work is a correct answer, so **paper hit@k is the primary metric**. Page hit is printed too
  but is strict: it counts only the page(s) recorded in the file, so it understates real accuracy.
- **exact_term**: the question hinges on a rare exact string (gene name, SNP ID, catalog number, tool name),
  written to test keyword matching; each term occurs in exactly one paper. Scored like disambiguation (paper hit
  primary; some terms appear on many pages, so page hit is strict and secondary).
- **multi_document**: summary-style questions spanning several papers. Reported: file recall@k, the fraction
  of expected papers present in the top k chunks.
- **out_of_corpus**: questions the papers cannot answer. With `--refusal`, an LLM judge checks the assistant
  declines instead of inventing an answer.

Definitions: *hit@k* = the right paper (or page) is among the top k chunks; *MRR* = mean of 1/rank of the
first right chunk (1.0 = always first).

## Answer-level check (`--answers`)

Generates the real answer to each multi-paper question at each k and reports how many expected papers were
retrieved and how many the answer cites by file name (`--show` prints the answers). The citation count is a
rough signal only: the model sometimes cites author-year entries copied from a paper's reference list instead
of the file name, which this count misses. Read the answers.

## Context precision and answer-level checks

- **Precision@k** is printed with each category: the share of the top-k chunks that come from the right paper
  (and, for specific questions, strictly on the right page). It shows how much noise a larger k adds.
- **`--judge`** generates the real answer to every answerable question at `--answer-k` (default `RETRIEVAL_K`) and reports:
  - *faithfulness*: an LLM judge (`--judge-model`, temperature 0) splits the answer into claims and marks each
    supported or not by the retrieved context; the score is the supported fraction;
  - *relevancy*: a 1-5 rating of how directly the answer addresses the question;
  - *citations* (deterministic): of the citations in the answer text, how many name a paper that was retrieved
    (*grounded*), cite pages that were retrieved (*page-correct*), refer to outside works by author-year
    (*outside*), name no retrieved paper (*unmatched*), or give a page with no file (*unnamed*).
  Run it with `--show` to read every answer and each unsupported claim.
- Judge scores hit the ceiling here, so read the answers and use the citation counts, which are the discriminating signal.

## Second corpus: documentation instead of papers

`eval/spring_questions.jsonl` (42 questions) targets the Spring Boot 3.1.12 reference, a very different document type:
code-heavy, hierarchical, with a table of contents and a properties appendix. Build the corpus and run it:

```bash
python -m eval.spring_docs ./spring-docs                      # downloads the official PDF and splits it into 112 files
COLLECTION_NAME=spring_docs DOCS_DIR=./spring-docs/docs python ingest.py
COLLECTION_NAME=spring_docs python -m eval.run_eval --questions eval/spring_questions.jsonl --strategy similarity
```

Extra conventions here: `expected_locations` lists every place a fact is documented (a narrative section *and* the
property appendix; any one counts), and the `paraphrased` category uses everyday wording with none of the docs'
vocabulary. Multi-section topic sets (`multi_document`) were curated by reading mention counts, since a dependency
table mentioning a library 49 times is not coverage of that library. The table of contents is deliberately left in
the index as a realistic noise source.

## Comparing chat models

`--judge`, `--answers` and `--refusal` generate answers with the configured chat model (`CHAT_PROVIDER` / `CHAT_MODEL`),
so the same questions can be run against Claude and OpenAI. The faithfulness, relevancy and refusal judges are always
`gpt-4o-mini` (`--judge-model`), independent of the answer model whenever Claude is answering. Retrieval metrics don't
depend on the chat model.

## Caveats

- Small sets (42 questions each): one question moves a category by several points.
- Questions were written from the same text they test, so scores lean optimistic.
- Answer quality is checked more weakly than retrieval: judge scores are near the ceiling, the judge shares a model family with the generator, and only the citation check is deterministic.
