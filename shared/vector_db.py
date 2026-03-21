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

    def _ensure_index(self):
        expected_dim = self._DIMENSIONS[self.provider]
        existing = {idx.name: idx for idx in self.pinecone.list_indexes()}

        if self.index_name in existing:
            current_dim = existing[self.index_name].dimension
            if current_dim != expected_dim:
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
        """Check if documents with a given identifier exist."""
        dummy_vector = [0.0] * self._DIMENSIONS[self.provider]
        results = self.index.query(
            vector=dummy_vector,
            top_k=1,
            filter={filter_key: {"$eq": identifier}},
            include_metadata=False,
        )
        return bool(results.matches)

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
        """Chunk and upsert financial reports for a ticker."""
        from datetime import datetime, timezone
        fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            self.index.delete(filter={"ticker": {"$eq": ticker}})
            logger.info("Deleted existing chunks for %s", ticker)
        except Exception as e:
            logger.warning("Failed to delete existing chunks for %s (ignored): %s", ticker, e)

        total_chunks = 0
        for report in reports_data:
            doc_id = f"{ticker}_{report['period']}_{report['type'].replace(' ', '_')}"
            metadata = {
                "ticker": ticker,
                "title": report["title"],
                "period": report["period"],
                "type": report["type"],
                "fetched_at": fetched_at,
            }
            total_chunks += self.upsert_chunks(doc_id, report["content"], metadata)
        logger.info("Upserted %d chunks for ticker='%s' (fetched_at=%s)", total_chunks, ticker, fetched_at)

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

    def upsert_papers(self, papers: list[dict[str, Any]]):
        """Convert PDFs to markdown, chunk, embed, and upsert into Pinecone."""
        total_chunks = 0
        for paper in papers:
            paper_id = self._paper_id(paper["pdf_path"])
            markdown = self.pdf_to_markdown(paper["pdf_path"])
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
            total_chunks += self.upsert_chunks(paper_id, markdown, metadata)
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
