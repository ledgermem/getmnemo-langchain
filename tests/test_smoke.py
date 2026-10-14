"""Adapter tests with a mocked getmnemo client (no network).

``Mnemo.search`` returns ``list[SearchHit]`` since getmnemo 0.2; these tests
pin that the adapter consumes it correctly and threads ``container_tag``
through every SDK call.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from getmnemo import Mnemo, SearchHit

from langchain_getmnemo import MnemoMemory, MnemoRetriever


def _hit(content: str, metadata: dict[str, Any] | None = None, **overrides: Any) -> SearchHit:
    payload: dict[str, Any] = {
        "resultType": "memory",
        "score": 0.9,
        "scopeKey": "user:jane",
        "content": content,
        "metadata": metadata,
        "memoryId": "11111111-1111-4111-8111-111111111111",
    }
    payload.update(overrides)
    return SearchHit.model_validate(payload)


def _client() -> Mnemo:
    client = Mnemo.__new__(Mnemo)  # skip __init__: no api key / httpx client needed
    client.search = MagicMock(return_value=[])  # type: ignore[method-assign]
    client.add = MagicMock(return_value=None)  # type: ignore[method-assign]
    client.delete = MagicMock(return_value=None)  # type: ignore[method-assign]
    client.list = MagicMock()  # type: ignore[method-assign]
    return client


def test_imports_resolve() -> None:
    assert MnemoMemory is not None
    assert MnemoRetriever is not None


def test_memory_save_and_load_threads_container() -> None:
    client = _client()
    memory = MnemoMemory(client=client, container_tag="user:jane")

    memory.save_context({"input": "hi"}, {"output": "hello"})
    assert client.add.call_count == 2
    _, first_kwargs = client.add.call_args_list[0]
    assert first_kwargs["container_tag"] == "user:jane"
    assert first_kwargs["metadata"]["role"] == "user"

    client.search.return_value = [
        _hit("hi there", metadata={"role": "assistant", "source": "langchain"})
    ]
    out = memory.load_memory_variables({"input": "hi"})
    assert out["history"] == "assistant: hi there"
    client.search.assert_called_once_with("hi", limit=5, container_tag="user:jane")


def test_memory_namespace_filters_hits() -> None:
    client = _client()
    client.search.return_value = [
        _hit("other thread", metadata={"role": "user", "namespace": "t2"}),
        _hit("mine", metadata={"role": "user", "namespace": "t1"}),
    ]
    memory = MnemoMemory(client=client, namespace="t1")
    out = memory.load_memory_variables({"input": "hi"})
    assert out["history"] == "user: mine"


def test_memory_empty_results_yield_empty_history() -> None:
    # getmnemo 0.2 returns [] both for empty containers (404 swallowed) and
    # abstained queries — the adapter must not blow up on it.
    memory = MnemoMemory(client=_client())
    out = memory.load_memory_variables({"input": "anything"})
    assert out == {"history": ""}


def test_retriever_returns_documents_with_metadata() -> None:
    client = _client()
    client.search.return_value = [
        _hit("abc", metadata={"k": "v"}, documentId="22222222-2222-4222-8222-222222222222")
    ]
    retriever = MnemoRetriever(client=client, top_k=3, container_tag="user:jane")
    docs = retriever.invoke("query")
    client.search.assert_called_once_with("query", limit=3, container_tag="user:jane")
    assert len(docs) == 1
    assert docs[0].page_content == "abc"
    assert docs[0].metadata["k"] == "v"
    assert docs[0].metadata["score"] == 0.9
    assert docs[0].metadata["memory_id"] == "11111111-1111-4111-8111-111111111111"
    assert docs[0].metadata["document_id"] == "22222222-2222-4222-8222-222222222222"


def test_retriever_defaults_to_client_container() -> None:
    client = _client()
    retriever = MnemoRetriever(client=client, top_k=2)
    retriever.invoke("query")
    client.search.assert_called_once_with("query", limit=2, container_tag=None)


def test_clear_snapshots_ids_and_honours_namespace() -> None:
    client = _client()
    page1 = MagicMock()
    page1.items = [
        MagicMock(id="m1", metadata={"namespace": "t1"}),
        MagicMock(id="m2", metadata={"namespace": "t2"}),
    ]
    page1.next_cursor = "cur"
    page2 = MagicMock()
    page2.items = [MagicMock(id="m3", metadata={"namespace": "t1"})]
    page2.next_cursor = None
    client.list.side_effect = [page1, page2]

    memory = MnemoMemory(client=client, namespace="t1", container_tag="user:jane")
    memory.clear()

    deleted = [call.args[0] for call in client.delete.call_args_list]
    assert deleted == ["m1", "m3"]
    for call in client.delete.call_args_list:
        assert call.kwargs["container_tag"] == "user:jane"
