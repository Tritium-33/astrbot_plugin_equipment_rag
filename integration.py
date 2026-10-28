"""One native retrieval call with corpus-supported spelling expansion."""

import asyncio
import inspect
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from .catalogue import DEFAULT_ALIASES, Query, query_references
from .neighbours import extend
from .storage import Catalogues

GUIDANCE_MARKER = "[装备检索说明]"
GUIDANCE = (
    GUIDANCE_MARKER
    + "检索保留完整问题、组合表达式、数字、改型后缀与年代限定。查询写法信息仅用于查找文献；完整写法出现过不证明独立型号，组成项不代表并列对象。依据原文及同文档连续段落区分名称用法、实际安装、试装、拟议、未采用和比较关系，再归属参数与时期；前段交代对象而后段使用代指时结合阅读。"
)


def integer_option(config, name, default, minimum, maximum):
    try:
        value = int(config.get(name, default))
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(minimum, min(maximum, value))


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

    def sources(self, names=None):
        return {
            kb_id: helper.vec_db.document_storage.db_path
            for kb_id, helper in self.manager.kb_insts.items()
            if not helper.init_error and (names is None or helper.kb.kb_name in names)
        }

    async def read_catalogue(self, sources):
        try:
            return await asyncio.to_thread(
                self.catalogues.read, sources, tuple(self.manager.kb_insts)
            )
        except (OSError, sqlite3.Error) as error:
            self.logger.warning("型号词表暂不可读，沿用原生查询：%s", error)
            return None

    async def warm(self):
        if self.enabled() and (sources := self.sources()):
            await self.read_catalogue(sources)

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
            sources = self.sources(bound.arguments["kb_names"])
            question = bound.arguments["query"]
            aliases = self.config.get("aliases", DEFAULT_ALIASES)
            query = Query(question, ())
            if next(query_references(question, aliases), None) is not None:
                catalogue = await self.read_catalogue(sources)
                if catalogue is None:
                    return await original(*args, **kwargs)
                query = catalogue.expand(
                    question,
                    aliases,
                    integer_option(self.config, "max_aliases", 8, 1, 32),
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
                    for root in query.matches
                    for match in root.primary()
                ],
                "expressions": [match.as_dict() for match in query.matches],
            }
            context = output.get("context_text", "")
            if query.expression_context:
                context += "\n" + query.expression_context
            output["context_text"] = context
            if anchors:
                output["equipment_context"] = []
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
            radius = integer_option(self.config, "neighbor_chunks", 1, 0, 3)
            budget = integer_option(self.config, "context_budget", 4, 0, 32)
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
