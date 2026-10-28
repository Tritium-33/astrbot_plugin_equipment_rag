"""Read-only, change-aware catalogue for native knowledge-base chunks."""

import sqlite3
from collections import OrderedDict, defaultdict
from pathlib import Path
from threading import RLock
from typing import NamedTuple

from .catalogue import Catalogue


class Store(NamedTuple):
    identity: tuple[str, int]
    connection: sqlite3.Connection
    version: int | None
    catalogue: Catalogue | None
    documents: dict


class Catalogues:
    def __init__(self):
        self.lock = RLock()
        self.stores = {}
        self.combined = OrderedDict()

    def _discard(self, kb_id):
        previous = self.stores.pop(kb_id, None)
        if previous:
            previous.connection.close()
        self.combined.clear()

    @staticmethod
    def _documents(connection, kb_id, previous):
        stamps = defaultdict(list)
        for chunk_id, updated, doc_id in connection.execute(
            "SELECT doc_id, updated_at, json_extract(metadata, '$.kb_doc_id') "
            "FROM documents WHERE json_extract(metadata, '$.kb_id')=? ORDER BY doc_id",
            (kb_id,),
        ):
            stamps[doc_id].append((chunk_id, updated))
        documents = {}
        for doc_id, rows in stamps.items():
            stamp = tuple(rows)
            cached = previous.get(doc_id)
            if cached is None or cached[0] != stamp:
                cached = (
                    stamp,
                    Catalogue(
                        row[0]
                        for row in connection.execute(
                            "SELECT text FROM documents WHERE json_extract(metadata, '$.kb_id')=? "
                            "AND json_extract(metadata, '$.kb_doc_id') IS ?",
                            (kb_id, doc_id),
                        )
                    ),
                )
            documents[doc_id] = cached
        return documents

    def read(self, sources, active_ids=None):
        """sources maps kb_id to native DocumentStorage.db_path; never writes."""
        with self.lock:
            if active_ids is not None:
                for kb_id in self.stores.keys() - set(active_ids):
                    self._discard(kb_id)
            versions, catalogues = [], []
            for kb_id, path in sorted(sources.items()):
                path = Path(path).resolve()
                identity = (str(path), path.stat().st_ino)
                store = self.stores.get(kb_id)
                if store is None or store.identity != identity:
                    self._discard(kb_id)
                    connection = sqlite3.connect(
                        path.as_uri() + "?mode=ro", uri=True, check_same_thread=False
                    )
                    store = self.stores[kb_id] = Store(
                        identity, connection, None, None, {}
                    )
                connection = store.connection
                try:
                    current = connection.execute("PRAGMA data_version").fetchone()[0]
                    if current != store.version:
                        with connection:
                            connection.execute("BEGIN")
                            documents = self._documents(
                                connection, kb_id, store.documents
                            )
                        catalogue = Catalogue.merge(
                            entry[1] for entry in documents.values()
                        )
                        store = self.stores[kb_id] = store._replace(
                            version=current, catalogue=catalogue, documents=documents
                        )
                except Exception:
                    self._discard(kb_id)
                    raise
                versions.append((kb_id, identity, current))
                catalogues.append(store.catalogue)
            key = tuple(versions)
            if key not in self.combined:
                self.combined[key] = (
                    catalogues[0]
                    if len(catalogues) == 1
                    else Catalogue.merge(catalogues)
                )
            self.combined.move_to_end(key)
            if len(self.combined) > 4:
                self.combined.popitem(last=False)
            return self.combined[key]

    def close(self):
        with self.lock:
            for kb_id in tuple(self.stores):
                self._discard(kb_id)
            self.combined.clear()
