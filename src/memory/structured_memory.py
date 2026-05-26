"""Three-layer persistent memory: working -> episodic -> semantic (ChromaDB-backed)."""

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

logging.basicConfig(handlers=[RichHandler(rich_tracebacks=True)], level=logging.INFO)
logger = logging.getLogger(__name__)

_CHROMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "chroma")
_MODEL_NAME = "all-MiniLM-L6-v2"
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

    Layer 1 -- Working memory  : in-process deque of recent turn summaries (last 5).
    Layer 2 -- Episodic memory : ChromaDB collection; one document per turn.
    Layer 3 -- Semantic memory : ChromaDB collection; explicit [REMEMBER: ...] facts.

    Design notes:
    - SentenceTransformer is loaded ONCE in _init() and reused for all calls.
    - Embeddings are computed manually and passed as plain float lists to ChromaDB,
      so no embedding_function wrapper is needed or passed to get_or_create_collection.
    - If _init() fails for any reason, _available is set False permanently -- no retry,
      no re-loading the model on every subsequent call.
    - SentenceTransformer is initialised with device='cpu' to avoid Windows meta-tensor
      errors from automatic CUDA detection.
    """

    def __init__(self, user_id: str = "default", collection_prefix: str = "assistant") -> None:
        self._user_id = user_id
        self._prefix = collection_prefix
        self._working: deque[str] = deque(maxlen=5)
        self._client: Any = None
        self._episodes: Any = None
        self._facts: Any = None
        self._model: Any = None       # SentenceTransformer instance, set in _init()
        self._ready: bool = False
        self._available: bool = True  # flips to False on permanent failure, no retry

    # -- private helpers -------------------------------------------------------

    def _encode(self, text: str) -> list[float]:
        """Encode a single string to a float list using the loaded sentence-transformer."""
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

    def _init(self) -> None:
        """Lazy initialisation -- called once, then guarded by _ready / _available."""
        if self._ready or not self._available:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb

            # Error 3 fix: explicit CPU device prevents Windows meta-tensor crash.
            self._model = SentenceTransformer(_MODEL_NAME, device="cpu")
            self._model = self._model.to("cpu")

            Path(_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=_CHROMA_PATH)

            # Error 2 fix: no embedding_function passed; we supply embeddings manually.
            self._episodes = self._client.get_or_create_collection(
                name=f"{self._prefix}_episodes",
            )
            self._facts = self._client.get_or_create_collection(
                name=f"{self._prefix}_facts",
            )
            self._ready = True
            logger.info(
                "StructuredMemoryManager ready | user=%s | episodes=%d | facts=%d",
                self._user_id,
                self._episodes.count(),
                self._facts.count(),
            )
        except Exception as exc:
            logger.error("StructuredMemoryManager init failed: %s", exc)
            # Error 4 fix: permanent failure -- never retry, never reload model.
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
            logger.warning("Episode store failed: %s", exc)

        # Layer 3 -- semantic facts from [REMEMBER: ...] tags
        facts = _REMEMBER_RE.findall(assistant_msg)
        for fact in facts:
            try:
                fact_text = fact.strip()
                fact_emb = self._encode(fact_text)
                self._facts.add(
                    documents=[fact_text],
                    ids=[str(uuid.uuid4())],
                    embeddings=[fact_emb],
                    metadatas=[{"user_id": self._user_id, "ts": time.time()}],
                )
                logger.info("Stored fact: %r", fact_text)
            except Exception as exc:
                logger.warning("Fact store failed: %s", exc)

    def get_context(self, query: str) -> str:
        """Return a formatted prompt block of relevant memories for the query."""
        self._init()
        ctx = MemoryContext(working=list(self._working))

        if not self._ready:
            return ctx.as_prompt_block()

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
            logger.warning("Episode retrieval failed: %s", exc)

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
            logger.warning("Fact retrieval failed: %s", exc)

        return ctx.as_prompt_block()

    def get_all_facts(self) -> dict[str, list[str]]:
        """Return all stored memories for display in the UI."""
        self._init()
        episodes: list[str] = list(self._working)
        facts: list[str] = []

        if self._ready:
            try:
                count = self._facts.count()
                if count > 0:
                    res = self._facts.get(where={"user_id": self._user_id})
                    facts = res.get("documents", [])
            except Exception as exc:
                logger.warning("get_all_facts failed: %s", exc)

        return {"episodes": episodes, "facts": facts}

    def clear_working(self) -> None:
        self._working.clear()

    def clear_all(self) -> None:
        self._working.clear()
        if not self._ready:
            return
        try:
            self._episodes.delete(where={"user_id": self._user_id})
        except Exception:
            pass
        try:
            self._facts.delete(where={"user_id": self._user_id})
        except Exception:
            pass
        logger.info("StructuredMemoryManager cleared | user=%s", self._user_id)
