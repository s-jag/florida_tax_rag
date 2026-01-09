"""Graph expansion for retrieval results using Neo4j traversal."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.graph.client import Neo4jClient
from src.graph.queries import (
    get_all_citations_for_chunk,
    get_cited_documents,
    get_citing_documents,
    get_interpretation_chain,
)

from .models import CitationContext, RetrievalResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GraphExpander:
    """Expands retrieval results using Neo4j graph traversal.

    This class enriches search results with context from the knowledge graph,
    including parent chunks, citation relationships, and related documents.
    """

    def __init__(self, neo4j_client: Neo4jClient):
        """Initialize the graph expander.

        Args:
            neo4j_client: Neo4j client for graph queries
        """
        self.neo4j = neo4j_client

    def get_parent_chunk(self, chunk_id: str) -> dict | None:
        """Get the parent chunk for context.

        Args:
            chunk_id: ID of the child chunk

        Returns:
            Dict with parent chunk data or None if no parent
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[:CHILD_OF]->(parent:Chunk)
        RETURN parent.id AS id, parent.text AS text, parent.ancestry AS ancestry,
               parent.citation AS citation
        """

        results = self.neo4j.run_query(query, {"chunk_id": chunk_id})

        if results:
            return results[0]
        return None

    def get_sibling_chunks(self, chunk_id: str, limit: int = 3) -> list[str]:
        """Get sibling chunk IDs (other children of the same parent).

        Args:
            chunk_id: ID of the chunk
            limit: Maximum number of siblings to return

        Returns:
            List of sibling chunk IDs
        """
        query = """
        MATCH (c:Chunk {id: $chunk_id})-[:CHILD_OF]->(parent:Chunk)
        MATCH (parent)<-[:CHILD_OF]-(sibling:Chunk)
        WHERE sibling.id <> $chunk_id
        RETURN sibling.id AS id
        LIMIT $limit
        """

        results = self.neo4j.run_query(query, {"chunk_id": chunk_id, "limit": limit})

        return [r["id"] for r in results]

    def get_citations_for_chunk(self, chunk_id: str) -> list[CitationContext]:
        """Get citation relationships originating from this chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            List of CitationContext objects
        """
        edges = get_all_citations_for_chunk(self.neo4j, chunk_id)

        return [
            CitationContext(
                target_doc_id=edge.target_id,
                target_citation=edge.citation_text,
                relation_type=edge.relation_type,
                context_snippet=None,  # Could query for context if needed
            )
            for edge in edges
        ]

    def get_related_documents(self, doc_id: str, limit: int = 5) -> list[str]:
        """Find documents related via citations (both citing and cited).

        Args:
            doc_id: Document ID
            limit: Maximum number of related documents

        Returns:
            List of related document IDs
        """
        related = set()

        # Get documents this one cites
        cited = get_cited_documents(self.neo4j, doc_id)
        related.update(d.id for d in cited)

        # Get documents that cite this one
        citing = get_citing_documents(self.neo4j, doc_id)
        related.update(d.id for d in citing)

        return list(related)[:limit]

    def expand_statute(self, doc_id: str) -> list[str]:
        """For a statute, find implementing rules and interpreting cases/TAAs.

        Args:
            doc_id: Statute document ID (e.g., 'statute:212.05')

        Returns:
            List of related document IDs
        """
        # Extract section number from doc_id
        if not doc_id.startswith("statute:"):
            return []

        section = doc_id.replace("statute:", "")
        chain = get_interpretation_chain(self.neo4j, section)

        if not chain:
            return []

        related = []
        for rule in chain.implementing_rules:
            related.append(rule.id)
        for case in chain.interpreting_cases:
            related.append(case.id)
        for taa in chain.interpreting_taas:
            related.append(taa.id)

        return related

    def expand_rule(self, doc_id: str) -> list[str]:
        """For a rule, find parent statute and citing cases.

        Args:
            doc_id: Rule document ID

        Returns:
            List of related document IDs
        """
        # Find statutes this rule implements
        query = """
        MATCH (r:Rule {id: $doc_id})-[:IMPLEMENTS|AUTHORITY]->(s:Statute)
        RETURN s.id AS id
        """

        results = self.neo4j.run_query(query, {"doc_id": doc_id})
        related = [r["id"] for r in results]

        # Also get citing documents
        citing = get_citing_documents(self.neo4j, doc_id)
        related.extend(d.id for d in citing)

        return related

    def expand_case(self, doc_id: str) -> list[str]:
        """For a case, find cited statutes and rules.

        Args:
            doc_id: Case document ID

        Returns:
            List of related document IDs
        """
        cited = get_cited_documents(self.neo4j, doc_id)
        return [d.id for d in cited]

    def get_related_chunks_for_doc(self, doc_id: str, limit: int = 3) -> list[str]:
        """Get chunk IDs from a related document.

        Args:
            doc_id: Document ID
            limit: Maximum number of chunks to return

        Returns:
            List of chunk IDs
        """
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.level = 'parent'
        RETURN c.id AS id
        LIMIT $limit
        """

        results = self.neo4j.run_query(query, {"doc_id": doc_id, "limit": limit})
        return [r["id"] for r in results]

    def expand_results(
        self,
        results: list[RetrievalResult],
        max_expansion: int = 5,
    ) -> list[RetrievalResult]:
        """Expand retrieval results with graph context.

        For each result:
        1. Get parent chunk if level == "child"
        2. Get citation context for the chunk
        3. Find related documents based on doc_type
        4. Add related chunk IDs and apply graph boost

        Args:
            results: List of retrieval results to expand
            max_expansion: Maximum number of related documents to find per result

        Returns:
            Expanded retrieval results
        """
        for result in results:
            try:
                # 1. Get parent chunk if this is a child
                if result.level == "child" and not result.parent_chunk_id:
                    parent = self.get_parent_chunk(result.chunk_id)
                    if parent:
                        result.parent_chunk_id = parent["id"]

                # 2. Get citation context
                citations = self.get_citations_for_chunk(result.chunk_id)
                result.citation_context = citations

                # 3. Find related documents based on doc_type
                related_doc_ids = []
                if result.doc_type == "statute":
                    related_doc_ids = self.expand_statute(result.doc_id)
                elif result.doc_type == "rule":
                    related_doc_ids = self.expand_rule(result.doc_id)
                elif result.doc_type == "case":
                    related_doc_ids = self.expand_case(result.doc_id)
                else:  # taa or other
                    related_doc_ids = self.get_related_documents(result.doc_id, limit=max_expansion)

                # 4. Get chunk IDs from related documents
                related_chunks = []
                for doc_id in related_doc_ids[:max_expansion]:
                    chunks = self.get_related_chunks_for_doc(doc_id, limit=1)
                    related_chunks.extend(chunks)

                result.related_chunk_ids = related_chunks[:max_expansion]

                # 5. Apply graph boost based on connections
                if citations:
                    result.graph_boost += 0.05 * min(len(citations), 5)
                if related_chunks:
                    result.graph_boost += 0.02 * min(len(related_chunks), 5)

            except Exception as e:
                logger.warning(f"Failed to expand chunk {result.chunk_id}: {e}")
                # Continue with other results

        return results
