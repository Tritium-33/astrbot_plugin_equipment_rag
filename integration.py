"""One native retrieval call with corpus-supported spelling expansion."""

import asyncio
import inspect
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from .catalogue import DEFAULT_ALIASES
from .neighbours import extend
from .storage import Catalogues

GUIDANCE_MARKER = "[装备检索说明]"
GUIDANCE = (
    GUIDANCE_MARKER
    + "检索保留原问题的完整名称、数字、改型后缀与年代限定；比较对象可分别检索。候选写法不证明型号相同，回答须按具体来源归属参数，可结合同文档连续段落理解代指。"
)


@dataclass
class Call:
    sources: dict
    rerank: bool
    links: dict = field(default_factory=dict)


class NativeIntegration:
    def __init__(self, manager, config, logger):
        self.manager, self.config, self.logger = manager, config, logger
        self.catalogues = Catalogues()
        self.active = False
        self.original = None
        self.wrapper = None
        self.state = ContextVar("equipment_retrieval", default=None)
        self.fusion_hooks = []

    def enabled(self):
        return self.active and self.config.get("enabled", True)

    def install(self):
        if self.active:
            return
        original = self.manager.retrieve
        signature = inspect.signature(original)
        if not {"query", "kb_names"} <= set(signature.parameters):
            raise TypeError("装备检索插件需要 AstrBot 4.29.3 原生知识库接口")

        @wraps(original)
        async def retrieve(*args, **kwargs):
            if not self.enabled():
                return await original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            names = bound.arguments["kb_names"]
            sources = {
                kb_id: helper.vec_db.document_storage.db_path
                for kb_id, helper in self.manager.kb_insts.items()
                if helper.kb.kb_name in names and not helper.init_error
            }
            try:
                catalogue = await asyncio.to_thread(
                    self.catalogues.read, sources, tuple(self.manager.kb_insts)
                )
            except (OSError, sqlite3.Error) as error:
                self.logger.warning("型号词表暂不可读，沿用原生查询：%s", error)
                return await original(*args, **kwargs)
            query = catalogue.expand(
                bound.arguments["query"],
                self.config.get("aliases", DEFAULT_ALIASES),
                max(1, min(32, int(self.config.get("max_aliases", 8)))),
            )
            bound.arguments["query"] = query.text
            self.hook_fusion()
            call = Call(
                sources,
                any(
                    helper.vec_db.rerank_provider is not None
                    for kb_id, helper in self.manager.kb_insts.items()
                    if kb_id in sources
                ),
            )
            token = self.state.set(call)
            try:
                result = await original(*bound.args, **bound.kwargs)
            finally:
                self.state.reset(token)
            if not result or not result.get("results"):
                return result
            returned = {
                (row.get("kb_id"), row.get("doc_id"), row.get("chunk_id"))
                for row in result["results"]
            }
            anchors = {}
            for key, seed in call.links.items():
                seed_key = (seed.kb_id, seed.doc_id, seed.chunk_id)
                if key in returned and seed_key not in returned:
                    anchors[seed_key] = seed
            if not query.matches and not anchors:
                return result
            output = dict(result)
            output["equipment_spelling"] = {
                "query": query.original,
                "candidates": [
                    {"written": match.written, "source_forms": list(match.candidates)}
                    for match in query.matches
                ],
            }
            if self.config.get("llm_guidance", True):
                output["context_text"] = (
                    result.get("context_text", "") + "\n" + GUIDANCE
                )
            if anchors:
                output["equipment_context"] = []
                context = output.get("context_text", "")
                for seed in anchors.values():
                    output["equipment_context"].append(
                        {
                            "kb_id": seed.kb_id,
                            "doc_id": seed.doc_id,
                            "chunk_id": seed.chunk_id,
                            "content": seed.content,
                        }
                    )
                    context += f"\n【同文档相邻原文：{seed.doc_id}，块{seed.chunk_index}；邻接不等于同一型号】\n{seed.content}\n"
                output["context_text"] = context
            return output

        self.original, self.wrapper = original, retrieve
        self.manager.retrieve = retrieve
        self.active = True

    def hook_fusion(self):
        fusion = getattr(
            getattr(self.manager, "retrieval_manager", None), "rank_fusion", None
        )
        if fusion is None or any(owner is fusion for owner, _, _ in self.fusion_hooks):
            return
        original = fusion.fuse

        @wraps(original)
        async def fuse(*args, **kwargs):
            candidates = await original(*args, **kwargs)
            call = self.state.get()
            if not self.enabled() or call is None or not call.rerank:
                return candidates
            radius = max(0, min(3, int(self.config.get("neighbor_chunks", 1))))
            budget = max(0, min(32, int(self.config.get("context_budget", 4))))
            try:
                output, call.links = await asyncio.to_thread(
                    extend, candidates, call.sources, radius, budget
                )
                return output
            except (OSError, sqlite3.Error) as error:
                self.logger.warning("相邻原文暂不可读，保留原生候选：%s", error)
                return candidates

        fusion.fuse = fuse
        self.fusion_hooks.append((fusion, original, fuse))

    def uninstall(self):
        self.active = False
        if self.manager.retrieve is self.wrapper:
            self.manager.retrieve = self.original
        for owner, original, wrapper in self.fusion_hooks:
            if owner.fuse is wrapper:
                owner.fuse = original
        self.fusion_hooks.clear()

    async def close(self):
        self.uninstall()
        await asyncio.to_thread(self.catalogues.close)
