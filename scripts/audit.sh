#!/usr/bin/env bash
# Known-vulnerability scan of the installed runtime dependencies (pip-audit, PyPI/OSV advisories).
# Run it in an environment with `pip install -r requirements.txt` done and pip-audit on the PATH.
#
# The advisories below are ACCEPTED, not ignored by accident: each is fixed only in LangChain 1.x
# (langchain-core 1.x, langchain-openai 1.x, langsmith 0.7+), a major migration this project has not
# done. Anything NOT listed here fails the script, so a new advisory is a signal to act.
# When the LangChain migration is done, delete this list.
set -euo pipefail

FROZEN="$(mktemp)"
trap 'rm -f "$FROZEN"' EXIT
python -m pip freeze --exclude-editable > "$FROZEN"

# accepted: langchain 0.3.30: 1 advisory(ies), fix needs a major upgrade
# accepted: langchain-anthropic 0.3.22: 1 advisory(ies), fix needs a major upgrade
# accepted: langchain-core 0.3.86: 2 advisory(ies), fix needs a major upgrade
# accepted: langchain-openai 0.3.16: 1 advisory(ies), fix needs a major upgrade
# accepted: langchain-text-splitters 0.3.11: 1 advisory(ies), fix needs a major upgrade
# accepted: langsmith 0.3.45: 4 advisory(ies), fix needs a major upgrade

pip-audit -r "$FROZEN" --no-deps --disable-pip \
  --ignore-vuln PYSEC-2026-2192 \
  --ignore-vuln PYSEC-2026-2556 \
  --ignore-vuln PYSEC-2026-2193 \
  --ignore-vuln PYSEC-2026-2562 \
  --ignore-vuln PYSEC-2026-76 \
  --ignore-vuln PYSEC-2026-77 \
  --ignore-vuln CVE-2026-59152 \
  --ignore-vuln PYSEC-2026-2555 \
  --ignore-vuln PYSEC-2026-2582 \
  --ignore-vuln PYSEC-2026-2583
