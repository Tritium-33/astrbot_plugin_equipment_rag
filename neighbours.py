"""Bounded adjacent source passages, without inferring equipment ownership."""

import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import replace
from pathlib import Path


def extend(candidates, sources, radius, budget):
    output = list(candidates)
    if not candidates or radius <= 0 or budget <= 0:
        return output, {}
    groups = defaultdict(list)
    for item in candidates:
        groups[(item.kb_id, item.doc_id)].append(item)
    windows = {}
    for (kb_id, doc_id), seeds in groups.items():
        if kb_id not in sources:
            continue
        indices = sorted(
            {
                i
                for seed in seeds
                for i in range(
                    max(0, seed.chunk_index - radius), seed.chunk_index + radius + 1
                )
            }
        )
        sql = (
            "SELECT doc_id, text, CAST(json_extract(metadata, '$.chunk_index') AS INTEGER) "
            "FROM documents WHERE json_extract(metadata, '$.kb_doc_id')=? "
            "AND json_extract(metadata, '$.kb_id')=? "
            "AND CAST(json_extract(metadata, '$.chunk_index') AS INTEGER) IN ("
            + ",".join("?" for _ in indices)
            + ")"
        )
        with closing(
            sqlite3.connect(
                Path(sources[kb_id]).resolve().as_uri() + "?mode=ro", uri=True
            )
        ) as connection:
            windows[(kb_id, doc_id)] = {
                index: (chunk_id, content)
                for chunk_id, content, index in connection.execute(
                    sql, [doc_id, kb_id, *indices]
                )
            }
    seen = {(r.kb_id, r.doc_id, r.chunk_index) for r in candidates}
    links = {}
    for distance in range(1, radius + 1):
        for seed in candidates:
            rows = windows.get((seed.kb_id, seed.doc_id), {})
            for index in (seed.chunk_index + distance, seed.chunk_index - distance):
                key = (seed.kb_id, seed.doc_id, index)
                if key in seen or index not in rows:
                    continue
                chunk_id, content = rows[index]
                output.append(
                    replace(seed, chunk_id=chunk_id, chunk_index=index, content=content)
                )
                seen.add(key)
                links[(seed.kb_id, seed.doc_id, chunk_id)] = seed
                if len(links) >= budget:
                    return output, links
    return output, links
