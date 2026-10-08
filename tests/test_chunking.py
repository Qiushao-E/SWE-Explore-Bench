from __future__ import annotations

import pytest

from explorers import chunking
from explorers.bm25 import BM25Explorer


@pytest.mark.parametrize(
    ("line_counts", "max_chunks", "expected_regions"),
    [
        ([1, 1, 1, 1, 1], 2, [(0, 1, 1), (1, 1, 1)]),
        ([5, 1, 1], 2, [(0, 1, 3), (0, 3, 5)]),
        ([8], 2, [(0, 1, 3), (0, 3, 5)]),
        ([0, -1, 1, 1, 1], 2, [(2, 1, 1), (3, 1, 1)]),
        ([1], 5, [(0, 1, 1)]),
    ],
    ids=["short-files", "file-tail", "inside-file", "empty-files", "below-limit"],
)
def test_chunk_repo_respects_limit(
    tmp_path, monkeypatch, line_counts, max_chunks, expected_regions
):
    paths = []
    for index, line_count in enumerate(line_counts):
        path = tmp_path / f"source_{index}.py"
        content = " \n\n" if line_count < 0 else "".join(
            f"value_{line} = {line}\n" for line in range(line_count)
        )
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    # File iteration order is independent of the shared chunk budget.
    monkeypatch.setattr(chunking, "iter_source_files", lambda _root: iter(paths))
    chunks = chunking.chunk_repo(
        tmp_path, chunk_size=3, chunk_overlap=1, max_chunks=max_chunks
    )

    assert [(chunk.path, chunk.start, chunk.end) for chunk in chunks] == [
        (paths[index].name, start, end) for index, start, end in expected_regions
    ]


def test_bm25_index_respects_chunk_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"source_{index}.py").write_text(
            f"parser_{index} = {index}\n", encoding="utf-8"
        )

    explorer = BM25Explorer(tmp_path, max_chunks=2)
    results = explorer.explore(instance_id="chunk-limit", query="parser_0", top_k=10)

    assert len(results) == 2
    assert len({result.regions[0].path for result in results}) == 2
