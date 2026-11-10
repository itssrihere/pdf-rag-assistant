# Security policy

This is a solo learning project, maintained on a best-effort basis (no response-time guarantees). Only the
latest commit on `main` is supported.

## Reporting a vulnerability

Please report security issues **privately** through GitHub: open the **Security** tab and choose
**Report a vulnerability**. Please don't open a public issue for something exploitable. Useful details: what you
did, what happened, the commit you tested, and your Python and OS versions.

## What is in scope

The code in this repository: ingestion of PDFs, retrieval, the chat loop, the evaluation tooling, and the Docker
setup. Examples: a crafted PDF that makes ingestion misbehave beyond failing or being skipped, secrets written to
logs, or unsafe handling of file paths.

## Known and accepted

These are documented and not worth a report:

- **Dependency advisories** listed in [`scripts/audit.sh`](scripts/audit.sh): 15 advisories in the LangChain
  family whose fixes exist only in LangChain 1.x, a migration that has not been done. A *new* advisory fails the
  build and is welcome news.
- **Not hardened for exposure.** It is a local command-line tool with no authentication, rate limiting or input
  limits; it connects to Postgres as the superuser and the container runs as root. See "Security notes" in the
  [README](README.md) before putting any of it behind a network service.
- **Prompt injection through document text** can mislead an answer. The model has no tools, so it cannot take
  actions.
- **Data leaves your machine:** chunks and questions are sent to the embedding and chat providers (OpenAI, or
  Anthropic if you enable it).
