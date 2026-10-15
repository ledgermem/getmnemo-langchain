# getmnemo-langchain

LangChain adapter for [Mnemo Memory](https://mnemohq.com) — drop-in conversational memory and retriever backed by the [getmnemo](https://pypi.org/project/getmnemo/) SDK.

## Install

```bash
pip install getmnemo-langchain
```

Requires `getmnemo>=0.2`.

## Quickstart

```python
from getmnemo import Mnemo
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from langchain_getmnemo import MnemoMemory

mem_client = Mnemo(api_key="prfly_live_...", default_container_tag="user:jane")
chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    memory=MnemoMemory(client=mem_client, top_k=5),
)

print(chain.invoke({"input": "Remember that my favourite framework is FastAPI."}))
print(chain.invoke({"input": "What is my favourite framework?"}))
```

Every call is scoped to a container — a `"<type>:<id>"` string such as
`"user:jane"`. Set it once on the client (`default_container_tag=`) or per
component (`container_tag=`):

```python
memory = MnemoMemory(client=mem_client, container_tag="user:jane")
```

## Retriever for RAG

```python
from langchain_getmnemo import MnemoRetriever

retriever = MnemoRetriever(client=mem_client, top_k=8)
docs = retriever.invoke("show me what I said about deployments")
for doc in docs:
    print(doc.metadata.get("score"), doc.page_content)
```

## What you get

- `MnemoMemory` — implements `langchain_core.memory.BaseMemory`. Auto-saves every user / AI turn and rehydrates the top-K relevant prior turns by semantic search. An optional `namespace` partitions threads that share one container.
- `MnemoRetriever` — implements `langchain_core.retrievers.BaseRetriever`. Returns LangChain `Document` objects with the memory's metadata plus `score`, `memory_id`, and (for document-derived hits) `document_id`.

## License

MIT — see [LICENSE](./LICENSE).
