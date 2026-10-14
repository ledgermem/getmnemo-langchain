"""Mnemo-backed conversational memory for LangChain."""

from __future__ import annotations

from typing import Any

from getmnemo import Mnemo
from langchain_core.memory import BaseMemory
from pydantic import Field, PrivateAttr


class MnemoMemory(BaseMemory):
    """Conversational memory backed by Mnemo.

    On every turn, ``save_context`` writes the user/AI exchange into Mnemo,
    and ``load_memory_variables`` retrieves the top-K relevant prior turns by
    semantic search.

    All calls are scoped to ``container_tag`` when set, otherwise to the
    client's ``default_container_tag``. Use one container per conversation
    thread or user (e.g. ``"user:jane"``); ``namespace`` additionally
    partitions threads that share one container.
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    top_k: int = Field(default=5, ge=1, le=50)
    container_tag: str | None = None
    namespace: str | None = None

    _client: Mnemo = PrivateAttr()

    def __init__(self, client: Mnemo, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def _format_metadata(self, role: str) -> dict[str, Any]:
        meta: dict[str, Any] = {"role": role, "source": "langchain"}
        if self.namespace:
            meta["namespace"] = self.namespace
        return meta

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw_query = inputs.get(self.input_key, "")
        # LangChain chains routinely pass non-string inputs (dicts for
        # multi-modal, BaseMessage subclasses, lists). Coerce defensively
        # — passing a dict straight to ``client.search`` raised
        # ``TypeError: query must be str`` deep inside the SDK only at
        # runtime, with no clue which chain step caused it.
        if isinstance(raw_query, str):
            query = raw_query
        elif raw_query is None:
            query = ""
        else:
            content = getattr(raw_query, "content", None)
            query = content if isinstance(content, str) else str(raw_query)
        query = query.strip()
        if not query:
            return {self.memory_key: ""}
        # When a namespace is configured we must filter retrieved hits to
        # that namespace, otherwise memories from sibling threads sharing
        # the container leak into the prompt. Over-fetch so the post-filter
        # window still contains top_k matches.
        fetch_limit = self.top_k * 4 if self.namespace else self.top_k
        hits = self._client.search(
            query, limit=fetch_limit, container_tag=self.container_tag
        )
        lines: list[str] = []
        for hit in hits:
            metadata = hit.metadata or {}
            if self.namespace and metadata.get("namespace") != self.namespace:
                continue
            role = metadata.get("role", "memory")
            lines.append(f"{role}: {hit.content}")
            if len(lines) >= self.top_k:
                break
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        user_text = inputs.get(self.input_key)
        ai_text = outputs.get(self.output_key)
        if user_text:
            self._client.add(
                str(user_text),
                metadata=self._format_metadata("user"),
                container_tag=self.container_tag,
            )
        if ai_text:
            self._client.add(
                str(ai_text),
                metadata=self._format_metadata("assistant"),
                container_tag=self.container_tag,
            )

    def clear(self) -> None:
        # Mnemo does not expose a container-wide wipe in the SDK; iterate
        # and delete. Snapshot ids first — deleting while paginating mutates
        # the underlying collection and either skips rows or loops forever
        # depending on whether the backend uses offset or keyset cursors.
        # Honour the configured namespace so clear() doesn't nuke other
        # threads sharing the same container.
        ids: list[str] = []
        cursor: str | None = None
        while True:
            page = self._client.list(
                limit=100, cursor=cursor, container_tag=self.container_tag
            )
            for item in page.items:
                meta = item.metadata or {}
                if self.namespace and meta.get("namespace") != self.namespace:
                    continue
                ids.append(item.id)
            cursor = page.next_cursor
            if not cursor:
                break
        for memory_id in ids:
            self._client.delete(memory_id, container_tag=self.container_tag)
