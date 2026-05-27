"""Three-layer persistent memory: working -> episodic -> semantic.

Backend is selected at runtime via config.USE_PINECONE:
  False (default) -- ChromaDB on local disk (development)
  True            -- Pinecone serverless index (HF Spaces / production)

The public interface (add_turn, get_context, get_all_facts, clear_working,
clear_all) is identical regardless of backend.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from config import config

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

_CHROMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "chroma")
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384          # output dimension of all-MiniLM-L6-v2
_REMEMBER_RE = re.compile(r"\[REMEMBER:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass
class MemoryContext:
    working: list[str] = field(default_factory=list)   # last N turn summaries
    episodes: list[str] = field(default_factory=list)  # retrieved episodic hits
    facts: list[str] = field(default_factory=list)     # retrieved semantic hits

    def as_prompt_block(self) -> str:
        lines: list[str] = []
        if self.facts:
            lines.append("Remembered facts:")
            lines.extend(f"  - {f}" for f in self.facts)
        if self.episodes:
            lines.append("Relevant past context:")
            lines.extend(f"  - {e}" for e in self.episodes)
        return "\n".join(lines)


class StructuredMemoryManager:
    """Manages three memory layers for a single user/assistant pair.

    Layer 1 -- Working memory  : in-process deque of recent turn summaries.
    Layer 2 -- Episodic memory : vector store; one document per turn.
    Layer 3 -- Semantic memory : vector store; explicit [REMEMBER: ...] facts.

    Backend selection (config.USE_PINECONE):
      False -- ChromaDB PersistentClient writing to data/chroma/ on disk.
      True  -- Pinecone serverless index; namespaced per user/assistant pair.

    Design invariants:
    - SentenceTransformer is loaded once in _init() and reused across all calls.
    - _available = False is set permanently on any init failure; no retry.
    - device='cpu' prevents Windows meta-tensor errors from automatic CUDA detection.
    """

    def __init__(
        self,
        user_id: str = "default",
        collection_prefix: str = "assistant",
        working_memory_size: int = 5,
    ) -> None:
        self._user_id = user_id
        self._prefix = collection_prefix
        self._working: deque[str] = deque(maxlen=working_memory_size)
        self._model: Any = None
        self._ready: bool = False
        self._available: bool = True
        self._backend: str = "chromadb"

        # ChromaDB attributes
        self._client: Any = None
        self._episodes: Any = None
        self._facts: Any = None

        # Pinecone attributes
        self._index: Any = None
        self._ns_episodes: str = f"episodes-{user_id}-{collection_prefix}"
        self._ns_facts: str = f"facts-{user_id}-{collection_prefix}"

    # -- embedding helper ------------------------------------------------------

    def _encode(self, text: str) -> list[float]:
        """Encode a single string to a float list using the loaded sentence-transformer."""
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

    # -- backend initialisation ------------------------------------------------

    def _init_chromadb(self) -> None:
        """Set up ChromaDB PersistentClient and two collections."""
        import chromadb

        Path(_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=_CHROMA_PATH)

        # No embedding_function passed; embeddings are supplied manually.
        self._episodes = self._client.get_or_create_collection(
            name=f"{self._prefix}_episodes",
        )
        self._facts = self._client.get_or_create_collection(
            name=f"{self._prefix}_facts",
        )
        logger.info(
            "ChromaDB backend ready | user=%s | episodes=%d | facts=%d",
            self._user_id,
            self._episodes.count(),
            self._facts.count(),
        )

    def _init_pinecone(self) -> None:
        """Connect to (or create) the Pinecone index for this assistant pair."""
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=config.PINECONE_API_KEY)

        existing_names = [idx.name for idx in pc.list_indexes().indexes]
        if config.PINECONE_INDEX_NAME not in existing_names:
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=_EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Pinecone index created: %s", config.PINECONE_INDEX_NAME)

        self._index = pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(
            "Pinecone backend ready | user=%s | ns_episodes=%s | ns_facts=%s",
            self._user_id,
            self._ns_episodes,
            self._ns_facts,
        )

    def _init(self) -> None:
        """Lazy initialisation -- called once, then guarded by _ready / _available."""
        if self._ready or not self._available:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_MODEL_NAME, device="cpu")
            self._model = self._model.to("cpu")

            if config.USE_PINECONE:
                self._init_pinecone()
                self._backend = "pinecone"
            else:
                self._init_chromadb()
                self._backend = "chromadb"

            self._ready = True
            logger.info(
                "StructuredMemoryManager ready | user=%s | backend=%s",
                self._user_id,
                self._backend,
            )
        except Exception as exc:
            logger.error("StructuredMemoryManager init failed: %s", exc)
            self._available = False
            self._ready = False

    # -- public API ------------------------------------------------------------

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Persist one conversation turn to episodic and semantic layers."""
        self._init()

        summary = f"User: {user_msg[:200]} | Assistant: {assistant_msg[:300]}"
        self._working.append(summary)

        if not self._ready:
            return

        if self._backend == "pinecone":
            self._add_turn_pinecone(summary, assistant_msg)
        else:
            self._add_turn_chromadb(summary, assistant_msg)

    def _add_turn_chromadb(self, summary: str, assistant_msg: str) -> None:
        # Layer 2 -- episodic
        try:
            embedding = self._encode(summary)
            self._episodes.add(
                documents=[summary],
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas=[{"user_id": self._user_id, "ts": time.time()}],
            )
        except Exception as exc:
            logger.warning("Episode store failed (chromadb): %s", exc)

        # Layer 3 -- semantic facts from [REMEMBER: ...] tags
        for fact in _REMEMBER_RE.findall(assistant_msg):
            try:
                fact_text = fact.strip()
                fact_emb = self._encode(fact_text)
                self._facts.add(
                    documents=[fact_text],
                    ids=[str(uuid.uuid4())],
                    embeddings=[fact_emb],
                    metadatas=[{"user_id": self._user_id, "ts": time.time()}],
                )
                logger.info("Stored fact (chromadb): %r", fact_text)
            except Exception as exc:
                logger.warning("Fact store failed (chromadb): %s", exc)

    def _add_turn_pinecone(self, summary: str, assistant_msg: str) -> None:
        # Layer 2 -- episodic
        try:
            embedding = self._encode(summary)
            self._index.upsert(
                vectors=[{
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "user_id": self._user_id,
                        "collection_prefix": self._prefix,
                        "text": summary,
                        "ts": time.time(),
                    },
                }],
                namespace=self._ns_episodes,
            )
        except Exception as exc:
            logger.warning("Episode store failed (pinecone): %s", exc)

        # Layer 3 -- semantic facts from [REMEMBER: ...] tags
        for fact in _REMEMBER_RE.findall(assistant_msg):
            try:
                fact_text = fact.strip()
                fact_emb = self._encode(fact_text)
                self._index.upsert(
                    vectors=[{
                        "id": str(uuid.uuid4()),
                        "values": fact_emb,
                        "metadata": {
                            "user_id": self._user_id,
                            "collection_prefix": self._prefix,
                            "text": fact_text,
                            "ts": time.time(),
                        },
                    }],
                    namespace=self._ns_facts,
                )
                logger.info("Stored fact (pinecone): %r", fact_text)
            except Exception as exc:
                logger.warning("Fact store failed (pinecone): %s", exc)

    def get_context(self, query: str) -> str:
        """Return a formatted prompt block of relevant memories for the query."""
        self._init()
        ctx = MemoryContext(working=list(self._working))

        if not self._ready:
            return ctx.as_prompt_block()

        if self._backend == "pinecone":
            self._fill_context_pinecone(query, ctx)
        else:
            self._fill_context_chromadb(query, ctx)

        return ctx.as_prompt_block()

    def _fill_context_chromadb(self, query: str, ctx: MemoryContext) -> None:
        query_emb = self._encode(query)

        try:
            ep_count = self._episodes.count()
            if ep_count > 0:
                ep_res = self._episodes.query(
                    query_embeddings=[query_emb],
                    n_results=min(3, ep_count),
                    where={"user_id": self._user_id},
                )
                if ep_res["documents"] and ep_res["documents"][0]:
                    ctx.episodes = ep_res["documents"][0]
        except Exception as exc:
            logger.warning("Episode retrieval failed (chromadb): %s", exc)

        try:
            fact_count = self._facts.count()
            if fact_count > 0:
                fact_res = self._facts.query(
                    query_embeddings=[query_emb],
                    n_results=min(5, fact_count),
                    where={"user_id": self._user_id},
                )
                if fact_res["documents"] and fact_res["documents"][0]:
                    ctx.facts = fact_res["documents"][0]
        except Exception as exc:
            logger.warning("Fact retrieval failed (chromadb): %s", exc)

    def _fill_context_pinecone(self, query: str, ctx: MemoryContext) -> None:
        query_emb = self._encode(query)

        try:
            ep_res = self._index.query(
                vector=query_emb,
                top_k=3,
                namespace=self._ns_episodes,
                include_metadata=True,
            )
            ctx.episodes = [
                m.metadata["text"]
                for m in ep_res.matches
                if m.metadata and "text" in m.metadata
            ]
        except Exception as exc:
            logger.warning("Episode retrieval failed (pinecone): %s", exc)

        try:
            fact_res = self._index.query(
                vector=query_emb,
                top_k=5,
                namespace=self._ns_facts,
                include_metadata=True,
            )
            ctx.facts = [
                m.metadata["text"]
                for m in fact_res.matches
                if m.metadata and "text" in m.metadata
            ]
        except Exception as exc:
            logger.warning("Fact retrieval failed (pinecone): %s", exc)

    def get_all_facts(self) -> dict[str, list[str]]:
        """Return all stored memories for display in the UI."""
        self._init()
        episodes: list[str] = list(self._working)
        facts: list[str] = []

        if not self._ready:
            return {"episodes": episodes, "facts": facts}

        if self._backend == "pinecone":
            facts = self._get_all_facts_pinecone()
        else:
            facts = self._get_all_facts_chromadb()

        return {"episodes": episodes, "facts": facts}

    def _get_all_facts_chromadb(self) -> list[str]:
        try:
            count = self._facts.count()
            if count > 0:
                res = self._facts.get(where={"user_id": self._user_id})
                return res.get("documents", [])
        except Exception as exc:
            logger.warning("get_all_facts failed (chromadb): %s", exc)
        return []

    def _get_all_facts_pinecone(self) -> list[str]:
        # Pinecone does not support listing without a query vector.
        # Use a zero vector with large top_k to approximate a full scan.
        try:
            zero_vec = [0.0] * _EMBEDDING_DIM
            res = self._index.query(
                vector=zero_vec,
                top_k=100,
                namespace=self._ns_facts,
                include_metadata=True,
            )
            return [
                m.metadata["text"]
                for m in res.matches
                if m.metadata and "text" in m.metadata
            ]
        except Exception as exc:
            logger.warning("get_all_facts failed (pinecone): %s", exc)
        return []

    def clear_working(self) -> None:
        self._working.clear()

    def clear_all(self) -> None:
        self._working.clear()
        if not self._ready:
            return

        if self._backend == "pinecone":
            self._clear_pinecone()
        else:
            self._clear_chromadb()

        logger.info(
            "StructuredMemoryManager cleared | user=%s | backend=%s",
            self._user_id,
            self._backend,
        )

    def _clear_chromadb(self) -> None:
        try:
            self._episodes.delete(where={"user_id": self._user_id})
        except Exception:
            pass
        try:
            self._facts.delete(where={"user_id": self._user_id})
        except Exception:
            pass

    def _clear_pinecone(self) -> None:
        try:
            self._index.delete(delete_all=True, namespace=self._ns_episodes)
        except Exception as exc:
            logger.warning("Pinecone episode clear failed: %s", exc)
        try:
            self._index.delete(delete_all=True, namespace=self._ns_facts)
        except Exception as exc:
            logger.warning("Pinecone facts clear failed: %s", exc)
