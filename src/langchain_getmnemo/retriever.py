"""Mnemo-backed retriever for LangChain RAG pipelines."""

from __future__ import annotations

from typing import Any

from getmnemo import Mnemo
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr


class MnemoRetriever(BaseRetriever):
    """LangChain ``BaseRetriever`` that delegates to ``Mnemo.search``.

    The search is scoped to ``container_tag`` when set, otherwise to the
    client's ``default_container_tag``.
    """

    top_k: int = Field(default=5, ge=1, le=50)
    container_tag: str | None = None

    _client: Mnemo = PrivateAttr()

    def __init__(self, client: Mnemo, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = client

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        hits = self._client.search(
            query, limit=self.top_k, container_tag=self.container_tag
        )
        documents: list[Document] = []
        for hit in hits:
            metadata = dict(hit.metadata or {})
            metadata["score"] = hit.score
            if hit.memory_id is not None:
                metadata["memory_id"] = hit.memory_id
            if hit.document_id is not None:
                metadata["document_id"] = hit.document_id
            documents.append(Document(page_content=hit.content, metadata=metadata))
        return documents
