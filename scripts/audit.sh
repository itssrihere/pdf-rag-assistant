#!/usr/bin/env bash
# Known-vulnerability scan of the installed runtime dependencies (pip-audit, PyPI/OSV advisories).
# Run it in an environment with `pip install -r requirements.txt` done and pip-audit on the PATH.
#
# Any known advisory fails the script. Fix it by upgrading the package. If one truly cannot be fixed yet, add
# `--ignore-vuln <ID>` to the pip-audit call below, with a comment saying why it is safe here and what enforces that
# (ideally a test), and remove it as soon as the fix is available.
set -euo pipefail

FROZEN="$(mktemp)"
trap 'rm -f "$FROZEN"' EXIT
python -m pip freeze --exclude-editable > "$FROZEN"

pip-audit -r "$FROZEN" --no-deps --disable-pip
