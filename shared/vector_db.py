import asyncio
import json
import logging
import os
from typing import Any

import pymupdf4llm
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("mcp_tool_servers.vector_db")

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_EXIST_THRESHOLD = 0.7
_UPSERT_BATCH = 100


class VectorDB:
    """Consolidated vector DB client, parameterized by index_name.

    Supports both 'financial-reports' and 'research-papers' indices.
    """

    _DIMENSIONS = {"azure": 3072}

    def __init__(self, index_name: str, provider: str = "azure"):
        self.provider = provider
        self.index_name = index_name
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        self.embeddings = OpenAIEmbeddings(
            base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
            api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
            model="text-embedding-3-large",
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        self._ensure_index()

    def _ensure_index(self, allow_recreation: bool = False):
        expected_dim = self._DIMENSIONS[self.provider]
        existing = {idx.name: idx for idx in self.pinecone.list_indexes()}

        if self.index_name in existing:
            current_dim = existing[self.index_name].dimension
            if current_dim != expected_dim:
                if not allow_recreation:
                    raise ValueError(
                        f"Index '{self.index_name}' exists with dimension {current_dim} "
                        f"but expected {expected_dim}. Set allow_recreation=True to delete "
                        f"and recreate, or delete the index manually."
                    )
                logger.error(
                    "Dimension mismatch for '%s' (current=%d, expected=%d) — deleting and recreating index",
                    self.index_name, current_dim, expected_dim,
                )
                self.pinecone.delete_index(self.index_name)
                existing.pop(self.index_name)

        if self.index_name not in existing:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=expected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pinecone.Index(self.index_name)

    # ---- Generic operations ----

    def check_identifier(self, identifier: str, filter_key: str = "ticker") -> bool:
        """Check if documents with a given identifier exist.

        Uses fetch() on a deterministic chunk ID rather than a dummy-vector query —
        avoids a 3072-dim vector round-trip (~100-200 ms) just for an existence check.
        The first chunk of any stored document is always keyed as '{identifier}_chunk_0'.
        """
        probe_id = f"{identifier}_chunk_0"
        result = self.index.fetch(ids=[probe_id])
        return bool(result.vectors)

    def upsert_chunks(self, doc_id_base: str, content: str, metadata: dict[str, Any]):
        """Chunk, embed, and upsert a single document."""
        chunks = self.splitter.split_text(content)
        vectors = self.embeddings.embed_documents(chunks)

        upsert_data = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            meta = {**metadata, "chunk_index": i, "text": chunk}
            upsert_data.append({
                "id": f"{doc_id_base}_chunk_{i}",
                "values": vector,
                "metadata": meta,
            })

        for i in range(0, len(upsert_data), _UPSERT_BATCH):
            self.index.upsert(vectors=upsert_data[i:i + _UPSERT_BATCH])

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5,
                 filter_key: str = "", filter_value: str = "") -> list[dict[str, Any]]:
        """Semantically search and return top_k matching chunks."""
        query_vector = self.embeddings.embed_query(query)

        filter_dict = None
        if filter_key and filter_value:
            filter_dict = {filter_key: {"$eq": filter_value}}

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
        )

        chunks = []
        for match in results.matches:
            meta = match.metadata or {}
            chunks.append({**meta, "score": match.score})
        return chunks

    # ---- Financial reports ----

    def upsert_reports(self, ticker: str, reports_data: list[dict[str, Any]]):
        """Chunk and upsert financial reports for a ticker.

        Uses a write-then-swap pattern: new chunks are written with a fresh
        fetched_at timestamp, and old chunks are only deleted after all new
        chunks have been successfully upserted.  This prevents data loss if
        the upsert fails partway through.
        """
        from datetime import datetime, timezone
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fetch the old fetched_at so we can selectively delete stale chunks later
        old_fetched_at = self.get_last_fetched(ticker)

        # Phase 1: Upsert all new chunks (tagged with the new fetched_at)
        total_chunks = 0
        new_ids: list[str] = []
        for report in reports_data:
            doc_id_base = f"{ticker}_{fetched_at}_{report['period']}_{report['type'].replace(' ', '_')}"
            metadata = {
                "ticker": ticker,
                "title": report["title"],
                "period": report["period"],
                "type": report["type"],
                "fetched_at": fetched_at,
            }
            chunks = self.splitter.split_text(report["content"])
            vectors = self.embeddings.embed_documents(chunks)
            upsert_data = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = f"{doc_id_base}_chunk_{i}"
                new_ids.append(chunk_id)
                meta = {**metadata, "chunk_index": i, "text": chunk}
                upsert_data.append({"id": chunk_id, "values": vector, "metadata": meta})
            for i in range(0, len(upsert_data), _UPSERT_BATCH):
                self.index.upsert(vectors=upsert_data[i:i + _UPSERT_BATCH])
            total_chunks += len(chunks)

        logger.info("Upserted %d new chunks for ticker='%s' (fetched_at=%s)", total_chunks, ticker, fetched_at)

        # Phase 2: Delete old chunks only after all new chunks are written
        if old_fetched_at and old_fetched_at != fetched_at:
            try:
                self.index.delete(filter={
                    "ticker": {"$eq": ticker},
                    "fetched_at": {"$eq": old_fetched_at},
                })
                logger.info("Deleted old chunks for %s (fetched_at=%s)", ticker, old_fetched_at)
            except Exception as e:
                logger.warning(
                    "Failed to delete old chunks for %s (old data may linger): %s",
                    ticker, e,
                )

    def reports_exist(self, ticker: str) -> bool:
        return self.check_identifier(ticker, filter_key="ticker")

    def get_last_fetched(self, ticker: str) -> str | None:
        """Return the fetched_at date string for the most recently stored reports, or None."""
        dummy_vector = [0.1] + [0.0] * (self._DIMENSIONS[self.provider] - 1)
        results = self.index.query(
            vector=dummy_vector,
            top_k=1,
            filter={"ticker": {"$eq": ticker}},
            include_metadata=True,
        )
        if results.matches:
            return results.matches[0].metadata.get("fetched_at")
        return None

    # ---- Research papers ----

    @staticmethod
    def _paper_id(pdf_path: str) -> str:
        return os.path.splitext(os.path.basename(pdf_path))[0]

    @staticmethod
    def pdf_to_markdown(pdf_path: str) -> str:
        return pymupdf4llm.to_markdown(pdf_path)

    async def upsert_papers(self, papers: list[dict[str, Any]]):
        """Convert PDFs to markdown, chunk, embed, and upsert into Pinecone.

        All PDF→markdown conversions run in parallel threads (CPU-bound via
        pymupdf4llm).  Each paper's embedding + upsert also runs in a thread
        so the event loop is never blocked.
        """
        loop = asyncio.get_running_loop()

        async def _process_paper(paper: dict[str, Any]) -> int:
            paper_id = self._paper_id(paper["pdf_path"])
            markdown = await loop.run_in_executor(None, self.pdf_to_markdown, paper["pdf_path"])
            authors = (
                ", ".join(paper["authors"])
                if isinstance(paper["authors"], list)
                else paper["authors"]
            )
            metadata = {
                "paper_id": paper_id,
                "title": paper["title"],
                "authors": authors,
                "summary": paper["summary"],
                "pdf_path": paper["pdf_path"],
                "pdf_url": paper["pdf_url"],
            }
            return await loop.run_in_executor(None, self.upsert_chunks, paper_id, markdown, metadata)

        chunk_counts = await asyncio.gather(*[_process_paper(p) for p in papers])
        total_chunks = sum(chunk_counts)
        logger.info("Upserted %d chunks for %d paper(s)", total_chunks, len(papers))

    def papers_exist(self, query: str) -> bool:
        """Check if relevant papers exist for a given query."""
        query_vector = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_vector,
            top_k=2,
            include_metadata=False,
        )
        return bool(results.matches) and results.matches[0].score >= _EXIST_THRESHOLD


class UserKnowledgeDB:
    """Semantic memory store: durable user facts extracted from sessions.
    Uses a dedicated 'user-knowledge' Pinecone index with user_id metadata filtering."""

    _DIMENSIONS = {"azure": 3072}
    _DEDUP_THRESHOLD = 0.88  # skip upsert if a similar fact already exists

    def __init__(self):
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings = OpenAIEmbeddings(
            base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
            api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
            model="text-embedding-3-large",
        )
        self._ensure_index()

    def _ensure_index(self):
        existing = {idx.name for idx in self.pinecone.list_indexes()}
        if "user-knowledge" not in existing:
            self.pinecone.create_index(
                name="user-knowledge",
                dimension=self._DIMENSIONS["azure"],
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Created Pinecone index 'user-knowledge'")
        self.index = self.pinecone.Index("user-knowledge")

    def retrieve(self, user_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve semantic memory facts for a user matching the query."""
        vec = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=vec,
            top_k=top_k,
            filter={"user_id": {"$eq": user_id}},
            include_metadata=True,
        )
        return [
            {"fact": m.metadata.get("fact", ""), "score": m.score}
            for m in results.matches
        ]

    def upsert_facts(self, user_id: str, facts: list[str]) -> int:
        """Upsert facts with deduplication. Returns number of new facts inserted."""
        import uuid
        upserted = 0
        for fact in facts:
            vec = self.embeddings.embed_query(fact)
            # Check for near-duplicate before upserting
            existing = self.index.query(
                vector=vec,
                top_k=1,
                filter={"user_id": {"$eq": user_id}},
                include_metadata=True,
            )
            if existing.matches and existing.matches[0].score >= self._DEDUP_THRESHOLD:
                logger.debug("Skipping duplicate semantic fact for user=%s: %.60s", user_id, fact)
                continue
            self.index.upsert(vectors=[{
                "id": f"{user_id}_{uuid.uuid4().hex[:12]}",
                "values": vec,
                "metadata": {"user_id": user_id, "fact": fact},
            }])
            upserted += 1
        logger.info("Upserted %d new semantic facts for user=%s", upserted, user_id)
        return upserted
