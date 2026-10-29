"""
Retrieval evaluation over eval/questions.jsonl, against the collection in COLLECTION_NAME.

  python -m eval.run_eval                 # hit@k / MRR / recall@k at k = 1, 4, 8, 16
  python -m eval.run_eval --ks 4 8        # choose cut-offs
  python -m eval.run_eval --strategy mmr  # diversified retrieval (max marginal relevance) instead of plain similarity
  python -m eval.run_eval --strategy hybrid   # BM25 keyword + vector search fused by RRF (also: bm25 = keyword only)
  python -m eval.run_eval --answers       # generate real answers for multi-paper questions at several k (LLM calls)
  python -m eval.run_eval --judge         # answer-level: citations, faithfulness, relevancy (LLM calls, a few cents)
  python -m eval.run_eval --refusal       # also check the out-of-corpus questions are declined (LLM calls)
  python -m eval.run_eval --json out.json # save per-question results

Categories (see eval/README.md for the scoring rules):
  specific        one right paper and page; file and page hit both reported
  disambiguation  "which study...?"; paper hit is primary, page hit is secondary (strict: one recorded page)
  exact_term      question hinges on a rare exact string (gene, SNP, catalog number); scored like disambiguation
  multi_document  several right papers, scored as file recall@k
  out_of_corpus   refusal check only
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import config
from eval import metrics
from llm import make_chat_model, reply_text
from ingest import build_store

QUESTIONS = Path(__file__).parent / "questions.jsonl"
PAGE_IS_SECONDARY = {"disambiguation", "exact_term"}  # answer may appear on several pages of the right paper


def load_questions(path: Path | str = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def question_locations(q: dict) -> dict[str, list[int]]:
    """Where the answer lives: {file: pages}. A question lists `expected_locations` when a fact is documented in
    several places (any of them counts), or the older single `expected_file` / `expected_pages` pair."""
    if q.get("expected_locations"):
        return {loc["file"]: loc["pages"] for loc in q["expected_locations"]}
    return {q["expected_file"]: q["expected_pages"]}


_hybrid_retrievers: dict = {}  # one keyword index per store, built on first use


def retrieve(store, question: str, k: int, strategy: str = "similarity",
             fetch_k: int = 50, lambda_mult: float = 0.5, bm25_weight: float = 1.0) -> list[metrics.Chunk]:
    """Top-k chunks as (file, 1-based page). MMR re-ranks the `fetch_k` nearest chunks to favour variety."""
    if strategy in ("hybrid", "bm25"):
        import hybrid
        if id(store) not in _hybrid_retrievers:  # build the keyword index once, not per call
            _hybrid_retrievers[id(store)] = hybrid.HybridRetriever.from_store(store, fetch_k)
        retriever = _hybrid_retrievers[id(store)]
        retriever.bm25_weight = bm25_weight
        docs = retriever.search(question, k, mode=strategy)
    elif strategy == "mmr":
        docs = store.max_marginal_relevance_search(question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    else:
        docs = store.similarity_search(question, k=k)
    chunks = []
    for doc in docs:
        page = doc.metadata.get("page")
        chunks.append((doc.metadata.get("source_file", "unknown"), page + 1 if isinstance(page, int) else None))
    return chunks


def score_question(q: dict, retrieved_by_k: dict[int, list[metrics.Chunk]], ks: list[int]) -> dict:
    """`retrieved_by_k[k]` is the ranked result of a search asking for k chunks."""
    if q["category"] == "multi_document":
        result = {}
        for k in ks:
            result[f"file_recall@{k}"] = metrics.file_recall_at_k(retrieved_by_k[k], q["expected_files"], k)
            result[f"file_prec@{k}"] = metrics.file_precision_at_k(retrieved_by_k[k], q["expected_files"], k)
        return result
    result = {}
    locations = question_locations(q)
    for k in ks:
        result[f"file_prec@{k}"] = metrics.file_precision_at_k(retrieved_by_k[k], list(locations), k)
        result[f"page_prec@{k}"] = metrics.page_precision_at_k_any(retrieved_by_k[k], locations, k)
        result[f"file_hit@{k}"] = metrics.hit_at_k(metrics.first_file_rank_any(retrieved_by_k[k], locations), k)
        result[f"page_hit@{k}"] = metrics.hit_at_k(metrics.first_page_rank_any(retrieved_by_k[k], locations), k)
    widest = retrieved_by_k[ks[-1]]  # rank-based numbers come from the largest search
    file_rank = metrics.first_file_rank_any(widest, locations)
    page_rank = metrics.first_page_rank_any(widest, locations)
    result.update({"file_rank": file_rank, "page_rank": page_rank,
                   "file_mrr": metrics.reciprocal_rank(file_rank), "page_mrr": metrics.reciprocal_rank(page_rank)})
    return result


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def check_refusals(questions: list[dict]) -> dict[str, bool]:
    """For out-of-corpus questions: does the assistant say the answer isn't in the documents?"""
    from langchain_core.prompts import ChatPromptTemplate

    import query

    retriever, llm = query.build_retriever_and_llm()
    judge_llm = make_chat_model("openai", "gpt-4o-mini")  # independent of the answer model
    judge = ChatPromptTemplate.from_messages([
        ("system", "Reply with exactly YES or NO. Does the answer below say that the information is not "
                   "available in the provided documents (i.e. it declines to answer)?"),
        ("human", "{answer}"),
    ])
    declined = {}
    for q in questions:
        docs = retriever.invoke(q["question"])
        answer = reply_text(llm.invoke(query.ANSWER_PROMPT.format_messages(
            history=[], context=query.format_context(docs), question=q["question"])))
        verdict = reply_text(judge_llm.invoke(judge.format_messages(answer=answer))).strip().upper()
        declined[q["id"]] = verdict.startswith("YES")
    return declined


def check_answers(questions: list[dict], ks: list[int], show: bool) -> None:
    """Generate the real answer to each multi-paper question at several k and count which expected
    papers it actually cites (the prompt asks the model to cite source files)."""
    import query

    retriever, llm = query.build_retriever_and_llm()
    store = retriever.vectorstore
    for q in (q for q in questions if q["category"] == "multi_document"):
        print(f"\n{q['id']}: {q['question']}   ({len(q['expected_files'])} expected papers)")
        for k in ks:
            docs = store.similarity_search(q["question"], k=k)
            answer = reply_text(llm.invoke(query.ANSWER_PROMPT.format_messages(
                history=[], context=query.format_context(docs), question=q["question"])))
            retrieved = {d.metadata.get("source_file") for d in docs} & set(q["expected_files"])
            cited = [f for f in q["expected_files"] if f.removesuffix(".pdf").lower() in answer.lower()]
            print(f"   k={k:<3} retrieved {len(retrieved)}/{len(q['expected_files'])} papers, "
                  f"answer cites {len(cited)}/{len(q['expected_files'])}, {len(answer.split())} words")
            if show:
                print("   ---\n" + "\n".join("   " + line for line in answer.splitlines()) + "\n   ---")


def evaluate_answers(questions: list[dict], k: int, judge_model: str, show: bool) -> dict:
    """Generates the real answer to every answerable question with the top-k chunks, then scores it:
    citations (deterministic), faithfulness and relevancy (LLM judge, temperature 0)."""
    import query
    from eval import judge

    retriever, llm = query.build_retriever_and_llm()
    store = retriever.vectorstore
    judge_llm = make_chat_model("openai", judge_model)
    out = {}
    for q in (q for q in questions if q["category"] != "out_of_corpus"):
        docs = store.similarity_search(q["question"], k=k)
        context = query.format_context(docs)
        answer = reply_text(llm.invoke(query.ANSWER_PROMPT.format_messages(
            history=[], context=context, question=q["question"])))
        retrieved = [(d.metadata.get("source_file"), d.metadata["page"] + 1 if isinstance(d.metadata.get("page"), int) else None)
                     for d in docs]
        out[q["id"]] = {
            "category": q["category"], "answer": answer,
            "citations": judge.check_citations(judge.parse_citations(answer), retrieved),
            "faithfulness": judge.judge_faithfulness(judge_llm, context, answer),
            "relevancy": judge.judge_relevancy(judge_llm, q["question"], answer),
        }
        if show:
            r = out[q["id"]]
            print(f"\n{q['id']} [{q['category']}] {q['question']}\n  faithfulness {r['faithfulness']['score']:.2f} "
                  f"({r['faithfulness']['claims']} claims), relevancy {r['relevancy']['score']}/5 ({r['relevancy']['reason']}), "
                  f"citations {r['citations']}")
            for claim in r["faithfulness"]["unsupported"]:
                print(f"    UNSUPPORTED: {claim}")

    print(f"\n== answer quality at k={k} (judge: {judge_model}; {len(out)} answers)")
    for category in dict.fromkeys(v["category"] for v in out.values()):
        rows = [v for v in out.values() if v["category"] == category]
        totals = {key: sum(r["citations"][key] for r in rows)
                  for key in ("total", "grounded", "page_correct", "outside", "unmatched", "unnamed")}
        cited_answers = sum(r["citations"]["total"] > 0 for r in rows)
        print(f"   {category:<15} n={len(rows):<2} faithfulness {mean(r['faithfulness']['score'] for r in rows):.2f}   "
              f"relevancy {mean(r['relevancy']['score'] for r in rows):.1f}/5   "
              f"{cited_answers}/{len(rows)} answers cite; {totals['grounded']}/{totals['total']} citations grounded, "
              f"{totals['page_correct']} page-correct, {totals['outside']} outside author-year, "
              f"{totals['unmatched']} unmatched, {totals['unnamed']} unnamed")
    return out


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--questions", default=str(QUESTIONS), help="question file (default: eval/questions.jsonl)")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--strategy", choices=["similarity", "mmr", "hybrid", "bm25"], default="similarity")
    parser.add_argument("--fetch-k", type=int, default=50, help="MMR: candidate pool size")
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="hybrid: weight of the keyword list in rank fusion")
    parser.add_argument("--lambda-mult", type=float, default=0.5, help="MMR: 1.0 = pure relevance, 0.0 = max variety")
    parser.add_argument("--answers", action="store_true", help="answer multi-paper questions at each k (LLM calls)")
    parser.add_argument("--show", action="store_true", help="with --answers: print the full answers")
    parser.add_argument("--judge", action="store_true", help="answer-level checks: citations, faithfulness, relevancy")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--answer-k", type=int, default=config.RETRIEVAL_K, help="chunks given to the model with --judge")
    parser.add_argument("--refusal", action="store_true", help="also test out-of-corpus questions (LLM calls)")
    parser.add_argument("--json", help="write per-question results to this file")
    args = parser.parse_args()
    ks = sorted(args.ks)

    questions = load_questions(args.questions)
    store = build_store()
    strategy = args.strategy + (f" (fetch_k={args.fetch_k}, lambda={args.lambda_mult})" if args.strategy == "mmr" else
                                f" (bm25 weight {args.bm25_weight})" if args.strategy == "hybrid" else "")
    print(f"Collection: {config.COLLECTION_NAME}   questions: {len(questions)}   k: {ks}   strategy: {strategy}\n")

    results = {}
    for q in questions:
        if q["category"] == "out_of_corpus":
            continue
        retrieved_by_k = {k: retrieve(store, q["question"], k, args.strategy, args.fetch_k, args.lambda_mult, args.bm25_weight) for k in ks}
        results[q["id"]] = score_question(q, retrieved_by_k, ks)

    by_category = defaultdict(list)
    for q in questions:
        if q["id"] in results:
            by_category[q["category"]].append(q["id"])

    for category, ids in by_category.items():
        print(f"== {category} ({len(ids)} questions)")
        if category == "multi_document":
            for k in ks:
                print(f"   file recall@{k:<3} {mean(results[i][f'file_recall@{k}'] for i in ids):.2f}   "
                      f"precision {mean(results[i][f'file_prec@{k}'] for i in ids):.2f}")
        elif category in PAGE_IS_SECONDARY:
            for k in ks:
                print(f"   @{k:<3} paper hit {mean(results[i][f'file_hit@{k}'] for i in ids):.2f}   "
                      f"(page hit {mean(results[i][f'page_hit@{k}'] for i in ids):.2f}, strict: one recorded page)")
            print("   precision@k (share of chunks from the right paper): "
                  + "  ".join(f"@{k} {mean(results[i][f'file_prec@{k}'] for i in ids):.2f}" for k in ks))
            print(f"   MRR  paper {mean(results[i]['file_mrr'] for i in ids):.2f}")
        else:
            for k in ks:
                print(f"   @{k:<3} file hit {mean(results[i][f'file_hit@{k}'] for i in ids):.2f}   "
                      f"page hit {mean(results[i][f'page_hit@{k}'] for i in ids):.2f}")
            print("   precision@k (share of chunks from the right paper / on the right page): "
                  + "  ".join(f"@{k} {mean(results[i][f'file_prec@{k}'] for i in ids):.2f}/"
                              f"{mean(results[i][f'page_prec@{k}'] for i in ids):.2f}" for k in ks))
            print(f"   MRR  file {mean(results[i]['file_mrr'] for i in ids):.2f}   "
                  f"page {mean(results[i]['page_mrr'] for i in ids):.2f}")
        print()

    page_scored = [i for c, ids in by_category.items() if c == "specific" for i in ids]
    misses = [i for i in page_scored if not metrics.hit_at_k(results[i]["page_rank"], ks[-1])]
    print(f"Specific-question page misses even at k={ks[-1]}: {', '.join(misses) or 'none'}")
    paper_misses = [i for c, ids in by_category.items() if c in PAGE_IS_SECONDARY for i in ids
                    if not metrics.hit_at_k(results[i]["file_rank"], ks[-1])]
    print(f"Disambiguation paper misses even at k={ks[-1]}: {', '.join(paper_misses) or 'none'}")

    if args.answers:
        check_answers(questions, ks, args.show)

    if args.judge:
        results["_answers"] = evaluate_answers(questions, args.answer_k, args.judge_model, args.show)

    if args.refusal:
        ooc = [q for q in questions if q["category"] == "out_of_corpus"]
        declined = check_refusals(ooc)
        print(f"\n== out_of_corpus: declined {sum(declined.values())}/{len(declined)}"
              + "".join(f"\n   {i}: {'declined' if d else 'ANSWERED'}" for i, d in declined.items()))
        results["_refusal"] = declined

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
