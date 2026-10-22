import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def local_modules_needed(entry: str = "main.py") -> set[str]:
    """Every top-level module of this repo that `entry` imports, directly or through other modules."""
    needed, queue = set(), [entry]
    while queue:
        name = queue.pop()
        if name in needed:
            continue
        needed.add(name)
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            imported = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in imported:
                candidate = module.split(".")[0] + ".py"
                if (ROOT / candidate).exists():
                    queue.append(candidate)
    return needed


def test_dockerfile_copies_every_module_the_app_imports():
    copied = set()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        match = re.match(r"COPY\s+(.+?)\s+\./?\s*$", line)
        if match:
            copied |= set(match.group(1).split())
    missing = local_modules_needed() - copied
    assert not missing, f"Dockerfile does not COPY {sorted(missing)}; the container would fail on import"
