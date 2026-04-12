import json
from difflib import SequenceMatcher

from astrbot.api import AstrBotConfig, llm_tool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.tool import ToolSet


@register("kb_plus", "shijinkarusui", "AstrBot 知识库增强检索插件", "0.2.0")
class KBPlusPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}

    @filter.command_group("kb")
    def kb(self):
        """知识库增强命令组"""

    @kb.command("list")
    async def kb_list(self, event: AstrMessageEvent) -> None:
        """列出知识库和文件。用法：/kb list [关键词]"""
        keyword = self._extract_command_payload(event.message_str, "list")
        result = await self._tool_kb_list_impl(keyword)
        event.set_result(event.plain_result(result))

    @kb.command("topk")
    async def kb_topk(self, event: AstrMessageEvent) -> None:
        """查看当前 kb 检索 top_k 配置。"""
        event.set_result(
            event.plain_result(
                f"当前默认 top_k：{self._get_default_top_k()}，最大 top_k：{self._get_max_top_k()}"
            )
        )

    @kb.command("ask")
    async def kb_ask(self, event: AstrMessageEvent) -> None:
        """精确知识库检索。用法：/kb ask [库名,文件名...] [问题]。

        当前实现改为把任务交给主对话模型，并只暴露 astr_plus_kb_* 工具。
        """
        payload = self._extract_command_payload(event.message_str, "ask")
        target_text, question = self._split_ask_payload(payload)
        if not question:
            event.set_result(
                event.plain_result("用法：/kb ask [库名,文件名……] [问题]")
            )
            return

        prompt = self._build_kb_ask_prompt(target_text, question)
        yield await self._request_kb_llm(event, prompt)

    @kb.command("free")
    async def kb_free(self, event: AstrMessageEvent) -> None:
        """自由知识库检索。用法：/kb free [问题]。

        当前实现改为把任务交给主对话模型，并只暴露 astr_plus_kb_* 工具。
        """
        question = self._extract_command_payload(event.message_str, "free")
        if not question:
            event.set_result(event.plain_result("用法：/kb free [问题]"))
            return

        prompt = self._build_kb_free_prompt(question)
        yield await self._request_kb_llm(event, prompt)

    @llm_tool("astr_plus_kb_list")
    async def astr_plus_kb_list(self, event: AstrMessageEvent, keyword: str = "") -> str:
        """列出内置知识库及其文件，并支持按库名或文件名模糊过滤。

        Args:
            keyword(string): 可选的过滤关键词，可以是库名、文件名或其片段。
        """
        return await self._tool_kb_list_impl(keyword)

    @llm_tool("astr_plus_kb_match")
    async def astr_plus_kb_match(self, event: AstrMessageEvent, target_text: str) -> str:
        """对知识库名和文件名进行模糊匹配，返回候选命中项。

        Args:
            target_text(string): 用户输入的库名、文件名或逗号分隔的多个目标。
        """
        match_data = await self._match_targets_structured(target_text)
        return json.dumps(match_data, ensure_ascii=False, indent=2)

    @llm_tool("astr_plus_kb_search")
    async def astr_plus_kb_search(
        self,
        event: AstrMessageEvent,
        query: str,
        kb_names_text: str = "",
        doc_names_text: str = "",
        top_k: int = 0,
    ) -> str:
        """对指定知识库或文件范围执行知识库检索，并返回结果摘要。

        Args:
            query(string): 用户要查询的问题。
            kb_names_text(string): 逗号分隔的知识库名称列表，不填则表示不限制知识库。
            doc_names_text(string): 逗号分隔的文件名称列表，不填则表示不限制文件名。
            top_k(number): 返回的结果数量，传 0 表示使用插件默认值。
        """
        kb_names = self._split_csv(kb_names_text)
        doc_names = self._split_csv(doc_names_text)
        resolved_top_k = self._resolve_top_k(top_k)
        return await self._tool_kb_search_impl(
            query,
            kb_names,
            doc_names,
            resolved_top_k,
        )

    async def _request_kb_llm(self, event: AstrMessageEvent, prompt: str):
        conversation = await self._get_or_create_conversation(event)
        tool_set = self._build_kb_only_tool_set()
        return event.request_llm(
            prompt=prompt,
            conversation=conversation,
            tool_set=tool_set,
            system_prompt=self._build_kb_system_prompt(),
        )

    def _get_default_top_k(self) -> int:
        value = self.config.get("default_top_k", 5)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 5
        return max(1, value)

    def _get_max_top_k(self) -> int:
        value = self.config.get("max_top_k", 10)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 10
        return max(1, value)

    def _get_strict_doc_chunk_limit(self) -> int:
        value = self.config.get("strict_doc_chunk_limit", 1000)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 1000
        return max(1, value)

    def _resolve_top_k(self, top_k: int | None) -> int:
        default_top_k = self._get_default_top_k()
        max_top_k = self._get_max_top_k()
        try:
            requested = int(top_k or 0)
        except (TypeError, ValueError):
            requested = 0
        if requested <= 0:
            requested = default_top_k
        return max(1, min(requested, max_top_k))

    def _build_kb_only_tool_set(self) -> ToolSet:
        tool_mgr = self.context.get_llm_tool_manager()
        tool_set = ToolSet()
        for tool_name in [
            "astr_plus_kb_list",
            "astr_plus_kb_match",
            "astr_plus_kb_search",
        ]:
            tool = tool_mgr.get_func(tool_name)
            if tool and getattr(tool, "active", True):
                tool_set.add_tool(tool)
        return tool_set

    async def _get_or_create_conversation(self, event: AstrMessageEvent):
        conv_mgr = self.context.conversation_manager
        umo = event.unified_msg_origin
        cid = await conv_mgr.get_curr_conversation_id(umo)
        if not cid:
            cid = await conv_mgr.new_conversation(umo, event.get_platform_id())
        conversation = await conv_mgr.get_conversation(umo, cid)
        if conversation:
            return conversation
        cid = await conv_mgr.new_conversation(umo, event.get_platform_id())
        conversation = await conv_mgr.get_conversation(umo, cid)
        if not conversation:
            raise RuntimeError("无法创建新的对话。")
        return conversation

    def _build_kb_system_prompt(self) -> str:
        prompt = (
            "你正在处理 kb 专用命令。"
            "本轮只允许使用 astr_plus_kb_list、astr_plus_kb_match、astr_plus_kb_search 这三个工具。"
            "不要调用 astrbot 原生知识库工具 astr_kb_search，也不要使用其他无关工具。"
            "如果用户指定了库名或文件名，先调用 astr_plus_kb_match 再决定检索范围。"
            "如果需要列出候选项，调用 astr_plus_kb_list。"
            "复杂问题可以多次调用 astr_plus_kb_search 缩小范围或补充证据。"
            "最后基于工具返回结果直接回答用户，不要暴露内部推理。"
        )
        if self.config.get("enable_multi_round_hint", True):
            prompt += "允许进行多轮工具调用；如果首轮信息不足，不要急于作答。"
        return prompt

    def _build_kb_ask_prompt(self, target_text: str, question: str) -> str:
        return (
            "用户正在使用 kb ask 命令。\n"
            f"目标输入: {target_text or '（未提供）'}\n"
            f"问题: {question}\n\n"
            "你必须只使用 astr_plus_kb_* 工具完成任务。"
            "先识别目标库/文件，再在对应范围内检索，然后给出最终答案。"
        )

    def _build_kb_free_prompt(self, question: str) -> str:
        return (
            "用户正在使用 kb free 命令。\n"
            f"问题: {question}\n\n"
            "你必须只使用 astr_plus_kb_* 工具完成任务。"
            "先自行判断应该检索哪些库或文件：必要时先列出或模糊匹配，再检索，最后回答。"
        )

    async def _tool_kb_list_impl(self, keyword: str = "") -> str:
        kbs = await self.context.kb_manager.list_kbs()
        if not kbs:
            return "当前没有可用知识库。"

        keyword_norm = self._normalize(keyword)
        lines = []
        matched_count = 0

        for kb in kbs:
            kb_helper = await self.context.kb_manager.get_kb(kb.kb_id)
            if not kb_helper:
                continue
            docs = await kb_helper.list_documents(limit=1000)
            doc_names = [doc.doc_name for doc in docs]

            if keyword_norm:
                kb_hit = self._is_match(kb.kb_name, keyword_norm)
                doc_hits = [name for name in doc_names if self._is_match(name, keyword_norm)]
                if not kb_hit and not doc_hits:
                    continue
                shown_docs = doc_names if kb_hit else doc_hits
            else:
                shown_docs = doc_names

            matched_count += 1
            lines.append(f"- 知识库：{kb.kb_name}（文件数：{len(doc_names)}）")
            if shown_docs:
                for doc_name in shown_docs:
                    lines.append(f"  - {doc_name}")
            else:
                lines.append("  - （暂无文件）")

        if not lines:
            return f"没有匹配到关键词“{keyword}”对应的知识库或文件。"

        title = "知识库列表"
        if keyword_norm:
            title += f"（关键词：{keyword}，命中知识库：{matched_count}）"
        return title + "\n" + "\n".join(lines)

    async def _tool_kb_search_impl(
        self,
        query: str,
        kb_names: list[str] | None = None,
        doc_names: list[str] | None = None,
        top_k: int = 0,
    ) -> str:
        query = (query or "").strip()
        kb_names = self._dedupe_preserve_order(kb_names or [])
        doc_names = self._dedupe_preserve_order(doc_names or [])
        top_k = self._resolve_top_k(top_k)

        if not query:
            return "query 不能为空。"

        if not kb_names:
            kb_names = await self._get_all_kb_names()
        if not kb_names:
            return "当前没有可用知识库。"

        if doc_names:
            strict_doc_results = await self._search_in_specific_docs(
                query=query,
                kb_names=kb_names,
                doc_names=doc_names,
                top_k=top_k,
            )
            if strict_doc_results:
                return self._format_search_results(
                    query=query,
                    kb_names=kb_names,
                    doc_names=doc_names,
                    results=strict_doc_results,
                )
            return "已定位到目标知识库，但指定文件中没有检索到相关内容。"

        try:
            result = await self.context.kb_manager.retrieve(
                query=query,
                kb_names=kb_names,
                top_m_final=max(top_k * 3, top_k),
            )
        except Exception as exc:
            logger.warning(f"知识库检索失败: {exc}")
            return f"知识库检索失败：{exc!s}"

        if not result or not result.get("results"):
            return "没有检索到相关内容。"

        return self._format_search_results(
            query=query,
            kb_names=kb_names,
            doc_names=[],
            results=result["results"][:top_k],
        )

    async def _search_in_specific_docs(
        self,
        query: str,
        kb_names: list[str],
        doc_names: list[str],
        top_k: int,
    ) -> list[dict]:
        doc_targets = await self._resolve_doc_targets(kb_names, doc_names)
        if not doc_targets:
            return []

        scored_chunks = []
        query_norm = self._normalize(query)
        query_tokens = self._tokenize_text(query)

        for target in doc_targets:
            kb_helper = target["kb_helper"]
            chunks = await kb_helper.get_chunks_by_doc_id(
                target["doc_id"],
                limit=self._get_strict_doc_chunk_limit(),
            )
            for chunk in chunks:
                content = chunk.get("content", "")
                score = self._score_chunk_text(content, query_norm, query_tokens)
                if score <= 0:
                    continue
                scored_chunks.append(
                    {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "doc_id": target["doc_id"],
                        "kb_id": target["kb_id"],
                        "kb_name": target["kb_name"],
                        "doc_name": target["doc_name"],
                        "chunk_index": chunk.get("chunk_index", 0),
                        "content": content,
                        "score": score,
                        "char_count": chunk.get("char_count", len(content)),
                    }
                )

        scored_chunks.sort(
            key=lambda item: (item["score"], -int(item.get("chunk_index", 0))),
            reverse=True,
        )
        return scored_chunks[:top_k]

    async def _resolve_doc_targets(
        self,
        kb_names: list[str],
        doc_names: list[str],
    ) -> list[dict]:
        targets = []
        normalized_doc_names = {self._normalize(name) for name in doc_names}
        for kb_name in kb_names:
            kb_helper = await self.context.kb_manager.get_kb_by_name(kb_name)
            if not kb_helper:
                continue
            docs = await kb_helper.list_documents(limit=1000)
            for doc in docs:
                if self._normalize(doc.doc_name) not in normalized_doc_names:
                    continue
                targets.append(
                    {
                        "kb_helper": kb_helper,
                        "kb_id": kb_helper.kb.kb_id,
                        "kb_name": kb_helper.kb.kb_name,
                        "doc_id": doc.doc_id,
                        "doc_name": doc.doc_name,
                    }
                )
        return targets

    def _format_search_results(
        self,
        query: str,
        kb_names: list[str],
        doc_names: list[str],
        results: list[dict],
    ) -> str:
        if not results:
            return "没有检索到相关内容。"
        lines = [f"检索问题：{query}"]
        lines.append(f"检索范围：{', '.join(kb_names)}")
        if doc_names:
            lines.append(f"限定文件：{', '.join(doc_names)}")
        lines.append("")

        for index, item in enumerate(results, 1):
            lines.append(f"【结果 {index}】")
            lines.append(f"来源：{item.get('kb_name', '')} / {item.get('doc_name', '')}")
            lines.append(f"相关度：{float(item.get('score', 0)):.4f}")
            lines.append(f"内容：{item.get('content', '')}")
            lines.append("")

        return "\n".join(lines).strip()

    def _score_chunk_text(
        self,
        content: str,
        query_norm: str,
        query_tokens: list[str],
    ) -> float:
        content_norm = self._normalize(content)
        if not content_norm:
            return 0.0

        score = 0.0
        if query_norm and query_norm in content_norm:
            score += 3.0

        for token in query_tokens:
            if token and token in content_norm:
                score += 1.2

        score += SequenceMatcher(None, content_norm[:500], query_norm).ratio()
        return score

    def _tokenize_text(self, text: str) -> list[str]:
        separators = ["，", ",", "；", ";", "？", "?", "。", ".", "\n", "\t"]
        normalized = text
        for separator in separators:
            normalized = normalized.replace(separator, " ")
        return [token for token in normalized.split() if token]

    async def _match_targets_structured(self, target_text: str) -> dict:
        tokens = self._split_csv(target_text)
        kbs = await self.context.kb_manager.list_kbs()
        kb_entries = []
        for kb in kbs:
            kb_helper = await self.context.kb_manager.get_kb(kb.kb_id)
            if not kb_helper:
                continue
            docs = await kb_helper.list_documents(limit=1000)
            kb_entries.append(
                {
                    "kb_name": kb.kb_name,
                    "doc_names": [doc.doc_name for doc in docs],
                }
            )

        if not tokens:
            return {
                "input": target_text,
                "tokens": [],
                "matched_kbs": [],
                "matched_docs": [],
                "unmatched": [],
            }

        matched_kbs = []
        matched_docs = []
        unmatched = []

        for token in tokens:
            kb_hits = []
            doc_hits = []
            token_norm = self._normalize(token)

            for entry in kb_entries:
                kb_name = entry["kb_name"]
                if self._is_match(kb_name, token_norm):
                    kb_hits.append(kb_name)
                for doc_name in entry["doc_names"]:
                    if self._is_match(doc_name, token_norm):
                        doc_hits.append({"kb_name": kb_name, "doc_name": doc_name})

            if kb_hits:
                matched_kbs.extend(kb_hits)
            if doc_hits:
                matched_docs.extend(doc_hits)
            if not kb_hits and not doc_hits:
                unmatched.append(token)

        return {
            "input": target_text,
            "tokens": tokens,
            "matched_kbs": self._dedupe_preserve_order(matched_kbs),
            "matched_docs": self._dedupe_dict_items(matched_docs),
            "unmatched": unmatched,
        }

    async def _get_all_kb_names(self) -> list[str]:
        kbs = await self.context.kb_manager.list_kbs()
        return [kb.kb_name for kb in kbs]

    def _split_ask_payload(self, payload: str) -> tuple[str, str]:
        payload = (payload or "").strip()
        if not payload:
            return "", ""
        if " " not in payload:
            return payload, ""
        target_text, question = payload.split(" ", 1)
        return target_text.strip(), question.strip()

    def _extract_command_payload(self, message_str: str, subcommand: str) -> str:
        text = (message_str or "").strip()
        prefix = f"/kb {subcommand}"
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
        prefix = f"kb {subcommand}"
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
        return text

    def _guess_targets_from_question(self, question: str) -> str:
        separators = ["，", ",", "；", ";", "？", "?", "。", "\n"]
        text = question
        for separator in separators:
            text = text.replace(separator, " ")
        words = [word.strip() for word in text.split(" ") if word.strip()]
        return ",".join(words[:8])

    def _split_csv(self, text: str) -> list[str]:
        if not text:
            return []
        normalized = (
            text.replace("，", ",")
            .replace("、", ",")
            .replace("；", ",")
            .replace(";", ",")
            .replace("|", ",")
        )
        return [part.strip() for part in normalized.split(",") if part.strip()]

    def _normalize(self, text: str) -> str:
        return "".join((text or "").lower().strip().split())

    def _is_match(self, candidate: str, keyword_norm: str) -> bool:
        candidate_norm = self._normalize(candidate)
        if not keyword_norm:
            return True
        if keyword_norm == candidate_norm:
            return True
        if keyword_norm in candidate_norm or candidate_norm in keyword_norm:
            return True
        return SequenceMatcher(None, candidate_norm, keyword_norm).ratio() >= 0.72

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _dedupe_dict_items(self, items: list[dict]) -> list[dict]:
        seen = set()
        result = []
        for item in items:
            key = (item.get("kb_name"), item.get("doc_name"))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
