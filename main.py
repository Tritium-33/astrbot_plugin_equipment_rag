from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

from .catalogue import DEFAULT_ALIASES, query_references
from .integration import GUIDANCE, GUIDANCE_MARKER, NativeIntegration


class EquipmentRAGPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.integration = NativeIntegration(context.kb_manager, config, self.logger)

    async def initialize(self):
        self.integration.install()
        self.logger.info("装备检索已启用语料写法匹配；检索与Rerank沿用原生配置。")

    @filter.on_llm_request(priority=10)
    async def guide_search(self, event: AstrMessageEvent, req: ProviderRequest):
        config = self.integration.config
        if not self.integration.enabled() or not config.get("llm_guidance", True):
            return
        question = event.get_message_str() or req.prompt or ""
        if next(
            query_references(question, config.get("aliases", DEFAULT_ALIASES)), None
        ) and GUIDANCE_MARKER not in (req.system_prompt or ""):
            req.system_prompt = (req.system_prompt or "") + "\n" + GUIDANCE

    async def terminate(self):
        await self.integration.close()
