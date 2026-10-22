"""
Single entry point: ingest PDFs from a folder, then start interactive Q&A.
Usage: python main.py --docs-dir /path/to/pdfs [--reset]
"""

import argparse

import config
from ingest import ingest
from query import build_retriever_and_llm, interactive_loop


def main():
    parser = argparse.ArgumentParser(
        description="RAG PDF — ingest a folder of PDFs and query them interactively"
    )
    parser.add_argument(
        "--docs-dir",
        default=config.DOCS_DIR,
        required=config.DOCS_DIR is None,
        help="Path to the folder containing PDF files",
    )
    parser.add_argument(
        "--keep-references",
        action="store_true",
        help="Embed bibliographies too (dropped by default)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Process at most this many PDFs, in path order (default: MAX_DOCS, 500; 0 = no limit)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before ingesting",
    )
    args = parser.parse_args()

    print(f"\nScanning {args.docs_dir} for PDFs...")
    result = ingest(args.docs_dir, reset=args.reset, keep_references=args.keep_references, max_docs=args.max_docs)
    if result.failed or result.not_attempted:
        print(f"\nNote: {len(result.failed)} PDF(s) failed and {result.not_attempted} were not attempted (see above); "
              "answers will only cover the documents that were stored.")

    print("\nConnecting to vector store...")
    retriever, llm = build_retriever_and_llm()
    interactive_loop(
        retriever,
        llm,
        banner=f"\nReady! I can answer questions about the docs in '{args.docs_dir}'.\nType 'exit' or 'quit' to stop.\n",
    )


if __name__ == "__main__":
    main()
