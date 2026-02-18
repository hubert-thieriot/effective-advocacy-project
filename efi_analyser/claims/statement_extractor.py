"""Extract statements from NER-enriched chunks using LLM."""

from __future__ import annotations

import json
import random
import logging
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from efi_analyser.scorers.openai_interface import OpenAIInterface
from efi_analyser.ner.types import NERResult, ChunkEntities, ConsolidatedNERResult
from .types import Statement, StatementsResult


class StatementExtractor:
    """Extract statements from chunks with identified actors using LLM.

    A statement is a claim or assertion made in a text chunk. The extractor
    identifies what is being said and who is saying it, linking back to
    NER-identified entities for speaker attribution.
    """

    def __init__(
        self,
        llm_client: OpenAIInterface,
        batch_size: int = 10,
        max_statements_per_chunk: int = 5,
        min_statement_length: int = 20,
    ) -> None:
        """Initialize statement extractor.

        Args:
            llm_client: OpenAI interface for LLM calls
            batch_size: Number of chunks to process per LLM call
            max_statements_per_chunk: Maximum statements to extract per chunk
            min_statement_length: Minimum statement length to include
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.max_statements_per_chunk = max_statements_per_chunk
        self.min_statement_length = min_statement_length
        self.logger = logging.getLogger(self.__class__.__name__)
        self.seed = 42
        self.rng = random.Random(self.seed)
        
    def extract(
        self,
        ner_result: NERResult,
        consolidated: Optional[ConsolidatedNERResult] = None,
        show_progress: bool = True,
        chunk_limit: Optional[int] = None,
    ) -> StatementsResult:
        """Extract statements from NER result chunks.

        Args:
            ner_result: NER extraction result with entities
            consolidated: Optional consolidated entities for speaker linking
            show_progress: Whether to show progress bar
            chunk_limit: Optional limit on number of chunks to process (for testing)

        Returns:
            StatementsResult with extracted statements
        """
        # Collect chunks that have entities (actors)
        chunks_with_entities = self._collect_chunks_with_entities(ner_result)

        if not chunks_with_entities:
            self.logger.warning("No chunks with entities found in NER result")
            return StatementsResult(
                statements=[],
                extraction_model=self.llm_client.spec_key(),
            )

        # Shuffle deterministically so that increasing chunk_limit only adds
        # new chunks at the end, preserving cached LLM results for earlier ones.
        shuffled = list(chunks_with_entities)
        self.rng.shuffle(shuffled)
        if chunk_limit is not None and chunk_limit < len(shuffled):
            self.logger.info(
                f"Limiting to {chunk_limit} chunks (of {len(shuffled)} with entities)"
            )
            shuffled = shuffled[:chunk_limit]
        chunks_with_entities = shuffled
        self.logger.info(
            f"Extracting statements from {len(chunks_with_entities)} chunks with entities"
        )

        # Build entity lookup for speaker linking
        entity_lookup = self._build_entity_lookup(consolidated)

        # Process in batches
        all_statements = self._process_batches(
            chunks_with_entities,
            entity_lookup,
            show_progress,
        )

        self.logger.info(f"Extracted {len(all_statements)} statements")

        return StatementsResult(
            statements=all_statements,
            extraction_model=self.llm_client.spec_key(),
        )

    def _collect_chunks_with_entities(
        self,
        ner_result: NERResult,
    ) -> List[ChunkEntities]:
        """Collect all chunks that have at least one entity."""
        chunks = []
        for doc in ner_result.documents:
            for chunk in doc.chunks:
                if chunk.entities:  # Only chunks with entities
                    chunks.append(chunk)
        return chunks

    def _build_entity_lookup(
        self,
        consolidated: Optional[ConsolidatedNERResult],
    ) -> Dict[str, str]:
        """Build lookup from entity name -> entity_id for speaker linking."""
        if consolidated is None:
            return {}

        lookup: Dict[str, str] = {}
        for entity in consolidated.entities:
            # Map canonical name
            lookup[entity.canonical_name.lower()] = entity.entity_id
            # Map all aliases
            for alias in entity.aliases:
                lookup[alias.lower()] = entity.entity_id

        return lookup

    def _process_batches(
        self,
        chunks: List[ChunkEntities],
        entity_lookup: Dict[str, str],
        show_progress: bool,
    ) -> List[Statement]:
        """Process chunks in batches."""
        all_statements: List[Statement] = []
        statement_counter = 1

        pbar = tqdm(
            total=len(chunks),
            desc=f"Extracting statements ({self.llm_client.spec_key()})",
            unit="chunk",
            disable=not show_progress,
        )

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                batch_statements = self._process_batch(batch, entity_lookup)
                # Assign statement IDs
                for stmt in batch_statements:
                    stmt.statement_id = f"stmt_{statement_counter:05d}"
                    statement_counter += 1
                all_statements.extend(batch_statements)
            except Exception as e:
                self.logger.error(f"Batch extraction failed: {e}")
                # Continue with next batch
            finally:
                pbar.update(len(batch))

        pbar.close()
        return all_statements

    def _process_batch(
        self,
        batch: List[ChunkEntities],
        entity_lookup: Dict[str, str],
    ) -> List[Statement]:
        """Process a batch of chunks through LLM."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(batch)

        try:
            response = self.llm_client.infer(system_prompt, user_prompt)
            statements = self._parse_response(response, batch, entity_lookup)
            return statements
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return []

    def _build_system_prompt(self) -> str:
        """Build system prompt for statement extraction."""
        return """You are an expert at extracting claims and statements from news text.

For each text chunk, identify statements that:
1. Express a claim, opinion, position, or assertion
2. Can be attributed to a speaker (quoted directly or paraphrased)
3. Are relevant to policy, debate, or public discourse

DO NOT extract:
- Simple factual descriptions without a stance
- Background information
- Statements that are too vague or generic

For each statement, identify:
- The statement text (what is being claimed)
- The speaker (who is making the claim) - use exact name as it appears
- The context (1-2 sentences of surrounding context)

Output valid JSON only."""

    def _build_user_prompt(self, batch: List[ChunkEntities]) -> str:
        """Build user prompt for a batch of chunks."""
        chunks_text = []
        for i, chunk in enumerate(batch):
            entity_names = [e.text for e in chunk.entities]
            chunks_text.append(
                f"Chunk {i+1} (ID: {chunk.chunk_id}):\n"
                f"Entities found: {', '.join(entity_names)}\n"
                f"Text: {chunk.text}\n"
            )

        prompt = f"""Extract statements from these text chunks.

{chr(10).join(chunks_text)}

For each chunk, extract up to {self.max_statements_per_chunk} statements.
Each statement should have:
- A clear claim/assertion (not just description)
- An identifiable speaker from the entities listed (or "unknown" if unclear)

Return JSON array:
[
  {{
    "chunk_id": "the chunk ID",
    "statements": [
      {{
        "text": "the statement/claim being made",
        "speaker": "name of the speaker (from entities) or 'unknown'",
        "context": "1-2 sentences of surrounding context"
      }}
    ]
  }}
]

If a chunk has no extractable statements, include it with an empty statements array."""
        return prompt

    def _parse_response(
        self,
        response: str,
        batch: List[ChunkEntities],
        entity_lookup: Dict[str, str],
    ) -> List[Statement]:
        """Parse LLM response into Statement objects."""
        try:
            # Clean response
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()

            data = json.loads(response_clean)
            if not isinstance(data, list):
                raise ValueError("Response must be a JSON array")

            # Build chunk lookup
            chunk_lookup = {chunk.chunk_id: chunk for chunk in batch}

            statements: List[Statement] = []
            for chunk_data in data:
                chunk_id = chunk_data.get("chunk_id", "")
                if chunk_id not in chunk_lookup:
                    continue

                chunk = chunk_lookup[chunk_id]
                # Extract doc_id from chunk_id (format: corpus@@doc_id:chunkNNN or doc_id:chunkNNN)
                doc_id = self._extract_doc_id(chunk_id)

                for stmt_data in chunk_data.get("statements", []):
                    text = stmt_data.get("text", "").strip()
                    if len(text) < self.min_statement_length:
                        continue

                    speaker = stmt_data.get("speaker", "").strip()
                    if speaker.lower() == "unknown":
                        speaker = None

                    # Look up speaker entity ID
                    speaker_entity_id = None
                    if speaker:
                        speaker_entity_id = entity_lookup.get(speaker.lower())

                    statements.append(
                        Statement(
                            statement_id="",  # Assigned later
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            text=text,
                            speaker=speaker,
                            speaker_entity_id=speaker_entity_id,
                            context=stmt_data.get("context", chunk.text[:200]),
                        )
                    )

            return statements

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return []

    def _extract_doc_id(self, chunk_id: str) -> str:
        """Extract document ID from chunk ID.

        Chunk IDs have format:
        - corpus@@doc_id:chunkNNN (multi-corpus)
        - doc_id:chunkNNN (single corpus)
        """
        # Remove chunk suffix
        if ":chunk" in chunk_id:
            base = chunk_id.split(":chunk")[0]
        else:
            base = chunk_id

        return base


__all__ = ["StatementExtractor"]
