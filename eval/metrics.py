"""Pure scoring functions for the retrieval evaluation (no network, no database).

A retrieved chunk is a (file name, 1-based page or None) tuple, ranked best-first."""

Chunk = tuple[str, int | None]


def first_file_rank(retrieved: list[Chunk], expected_file: str) -> int | None:
    """1-based rank of the first chunk from `expected_file`, or None if absent."""
    for rank, (name, _) in enumerate(retrieved, start=1):
        if name == expected_file:
            return rank
    return None


def first_page_rank(retrieved: list[Chunk], expected_file: str, expected_pages: list[int]) -> int | None:
    """1-based rank of the first chunk from `expected_file` on any of `expected_pages`."""
    for rank, (name, page) in enumerate(retrieved, start=1):
        if name == expected_file and page in expected_pages:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def file_recall_at_k(retrieved: list[Chunk], expected_files: list[str], k: int) -> float:
    """Fraction of `expected_files` that appear among the top-k chunks (for multi-paper questions)."""
    if not expected_files:
        return 0.0
    found = {name for name, _ in retrieved[:k]}
    return len(found & set(expected_files)) / len(expected_files)


def file_precision_at_k(retrieved: list[Chunk], expected_files: list[str], k: int) -> float:
    """Fraction of the top-k chunks that come from an expected paper (how much of the context is on topic)."""
    top = retrieved[:k]
    return sum(name in expected_files for name, _ in top) / len(top) if top else 0.0


def page_precision_at_k(retrieved: list[Chunk], expected_file: str, expected_pages: list[int], k: int) -> float:
    """Fraction of the top-k chunks from `expected_file` on one of `expected_pages` (strict: ignores other pages
    of the right paper that may also be relevant)."""
    top = retrieved[:k]
    return sum(name == expected_file and page in expected_pages for name, page in top) / len(top) if top else 0.0
