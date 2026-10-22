"""
Interactive Q&A over ingested PDF documents.
Usage: python query.py
       python query.py --question "What is the main topic?"
"""

import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

import config
from llm import make_chat_model, reply_text

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the conversation so far and a follow-up question, rewrite the follow-up as a "
        "standalone question that makes sense without the conversation. Replace pronouns and "
        "references with the names or terms they refer to. If it is already standalone, return "
        "it unchanged. Output only the question.",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant answering questions based on the provided documents. "
        "If the answer is not found in the documents, say so clearly — do not make up an answer. "
        "Answer the question concisely and cite the source file and page when possible.",
    ),
    MessagesPlaceholder("history"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def build_retriever_and_llm():
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
    store = PGVector(
        embeddings=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection=config.DB_URL,
        use_jsonb=True,
    )
    retriever = store.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
    return retriever, make_chat_model()


def _location(doc) -> tuple[str, int | None]:
    """(file name, 1-based page number or None) for a retrieved chunk."""
    page = doc.metadata.get("page")
    return doc.metadata.get("source_file", "unknown"), (page + 1 if isinstance(page, int) else None)


def format_context(docs: list) -> str:
    blocks = []
    for doc in docs:
        name, page = _location(doc)
        label = f"{name}, p. {page}" if page else name
        blocks.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_sources(docs: list) -> str:
    pages_by_file: dict[str, set[int]] = {}
    for doc in docs:
        name, page = _location(doc)
        pages = pages_by_file.setdefault(name, set())
        if page:
            pages.add(page)
    parts = []
    for name in sorted(pages_by_file):
        pages = sorted(pages_by_file[name])
        parts.append(f"{name} ({'p.' if len(pages) == 1 else 'pp.'} {', '.join(map(str, pages))})" if pages else name)
    return "; ".join(parts)


def ask(retriever, llm, question: str, history: list | None = None) -> str:
    """Answers `question`, using `history` (alternating Human/AI messages) for follow-ups."""
    history = history or []
    search_query = question
    if history:
        search_query = reply_text(llm.invoke(
            CONDENSE_PROMPT.format_messages(history=history, question=question)
        )).strip()

    docs = retriever.invoke(search_query)
    response = llm.invoke(
        ANSWER_PROMPT.format_messages(history=history, context=format_context(docs), question=question)
    )

    answer = reply_text(response)
    print(f"\nAnswer:\n{answer}")
    if docs:
        print(f"\nSources: {format_sources(docs)}")
    return answer


def trim_history(history: list, turns: int) -> list:
    """Keeps only the last `turns` Q&A pairs (two messages each); turns <= 0 disables memory."""
    return history[-2 * turns:] if turns > 0 else []


def interactive_loop(retriever, llm, banner: str = "RAG PDF Assistant — type 'exit' or 'quit' to stop\n") -> None:
    print(banner)
    history: list = []
    while True:
        try:
            question = input("Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            answer = ask(retriever, llm, question, history)
        except Exception as error:  # e.g. the OpenAI or Anthropic API is unreachable: keep the session alive
            print(f"\nSorry, that question failed ({type(error).__name__}: {' '.join(str(error).split())[:200]}). "
                  "Check your connection and try again.\n", file=sys.stderr)
            continue
        history += [HumanMessage(question), AIMessage(answer)]
        history = trim_history(history, config.HISTORY_TURNS)
        print()


def main():
    parser = argparse.ArgumentParser(description="Query ingested PDF documents")
    parser.add_argument("--question", "-q", help="Ask a single question and exit")
    args = parser.parse_args()

    print("Connecting to vector store...")
    retriever, llm = build_retriever_and_llm()

    if args.question:
        ask(retriever, llm, args.question)
    else:
        interactive_loop(retriever, llm)


if __name__ == "__main__":
    main()
