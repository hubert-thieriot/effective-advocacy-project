"""NER entity consolidation using LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import logging
import unicodedata

from tqdm import tqdm

from efi_analyser.scorers.openai_interface import OpenAIInterface
from .types import (
    NERResult,
    ConsolidatedEntity,
    ConsolidatedNERResult,
)


def normalize_entity_text(text: str) -> str:
    """Normalize entity text for matching.

    This handles unicode variations (different apostrophes, quotes, etc.)
    to ensure entities that should match do match.
    """
    # Normalize unicode to NFKC form (compatibility decomposition + composition)
    normalized = unicodedata.normalize("NFKC", text)

    # Replace common unicode variations with ASCII equivalents
    replacements = {
        '\u2018': "'",  # ' (left single quote)
        '\u2019': "'",  # ' (right single quote)
        '\u201c': '"',  # " (left double quote)
        '\u201d': '"',  # " (right double quote)
        '\u2013': "-",  # – (en dash)
        '\u2014': "-",  # — (em dash)
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    # Remove periods (for abbreviations like "F." vs "F")
    normalized = normalized.replace(".", "")

    # Normalize multiple spaces to single space
    normalized = " ".join(normalized.split())

    return normalized.lower().strip()


def normalize_display_name(name: str, entity_type: str) -> str:
    """Normalize entity name for display (fix all-caps, etc.).

    Args:
        name: The raw canonical name
        entity_type: Entity type (PERSON, ORG, etc.)

    Returns:
        Properly cased name for display
    """
    if not name:
        return name

    # Check if name is all caps (but not an acronym like "BBC", "NATO")
    # Consider it all-caps if it has 2+ words and all letters are uppercase
    words = name.split()
    is_all_caps = (
        len(words) >= 2
        and all(w.isupper() for w in words if w.isalpha())
    )

    if not is_all_caps:
        return name

    # For PERSON entities, always convert to title case
    if entity_type == "PERSON":
        # Handle names with special patterns (e.g., "McDonald", "O'Brien")
        result_words = []
        for word in words:
            lower = word.lower()
            # Handle Mc/Mac prefixes
            if lower.startswith("mc") and len(lower) > 2:
                result_words.append("Mc" + lower[2:].capitalize())
            elif lower.startswith("mac") and len(lower) > 3:
                result_words.append("Mac" + lower[3:].capitalize())
            # Handle O' prefixes
            elif lower.startswith("o'") and len(lower) > 2:
                result_words.append("O'" + lower[2:].capitalize())
            else:
                result_words.append(word.capitalize())
        return " ".join(result_words)

    # For ORG, be more careful - some orgs are legitimately all-caps
    # Only convert if it looks like a regular name (not an acronym)
    if entity_type == "ORG":
        # If any word is 3+ letters and all caps, leave as-is (might be intentional)
        for word in words:
            if len(word) >= 4 and word.isupper() and word.isalpha():
                # This looks like it should be title-cased
                return " ".join(w.capitalize() for w in words)
        # Otherwise, could be acronym-based, leave as-is
        return name

    # Default: title case
    return " ".join(w.capitalize() for w in words)


@dataclass
class EntityStats:
    """Statistics for a single raw entity."""

    text: str
    entity_type: str
    count: int
    frames: Dict[str, int]  # frame_id -> count


class NERConsolidator:
    """Consolidate raw NER entities using LLM."""

    def __init__(
        self,
        llm_client: OpenAIInterface,
        batch_size: int = 50,
        min_count: int = 2,
        entity_types: Optional[List[str]] = None,
        guidance: Optional[str] = None,
    ) -> None:
        """Initialize NER consolidator.

        Args:
            llm_client: OpenAI interface for LLM calls
            batch_size: Number of entities to send per LLM call
            min_count: Only consolidate entities appearing >= min_count times
            entity_types: Filter to specific entity types (e.g., ["PERSON", "ORG"])
            guidance: User-provided domain-specific guidance
        """
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.min_count = min_count
        self.entity_types = entity_types or ["PERSON", "ORG", "WORK_OF_ART", "EVENT", "FAC", "PRODUCT"]
        self.guidance = guidance
        self.logger = logging.getLogger(self.__class__.__name__)

    def consolidate(
        self,
        ner_result: NERResult,
        domain: Optional[str] = None,
    ) -> ConsolidatedNERResult:
        """Consolidate entities from NER result.

        Args:
            ner_result: Raw NER extraction result
            domain: Domain/topic context for LLM

        Returns:
            ConsolidatedNERResult with grouped entities
        """
        # 1. Aggregate entities
        entity_stats = self._aggregate_entities(ner_result)
        raw_count = len(entity_stats)
        self.logger.info(f"Aggregated {raw_count} unique entities from NER results")

        # 2. Filter by entity type
        entity_stats = [e for e in entity_stats if e.entity_type in self.entity_types]
        self.logger.info(f"Filtered to {len(entity_stats)} entities matching target types")

        # 3. Filter by min_count
        entity_stats = [e for e in entity_stats if e.count >= self.min_count]
        self.logger.info(f"Filtered to {len(entity_stats)} entities with count >= {self.min_count}")

        if not entity_stats:
            return ConsolidatedNERResult(
                entities=[],
                language=ner_result.language,
                frame_threshold=ner_result.frame_threshold,
                consolidation_model=self.llm_client.spec_key(),
                raw_entity_count=raw_count,
                consolidated_entity_count=0,
            )

        # 4. Batch LLM calls for consolidation
        consolidated_entities = self._llm_consolidate_all(entity_stats, domain or "")

        # 4b. Post-process: merge entities with identical (normalized_name, type)
        consolidated_entities = self._merge_duplicate_entities(consolidated_entities)

        # 5. Build type corrections map
        type_corrections = self._build_type_corrections(entity_stats, consolidated_entities)

        result = ConsolidatedNERResult(
            entities=consolidated_entities,
            language=ner_result.language,
            frame_threshold=ner_result.frame_threshold,
            consolidation_model=self.llm_client.spec_key(),
            raw_entity_count=raw_count,
            consolidated_entity_count=len(consolidated_entities),
            type_corrections=type_corrections,
        )

        self.logger.info(
            f"Consolidation complete: {raw_count} raw → {len(consolidated_entities)} consolidated entities"
        )
        return result

    def _aggregate_entities(self, ner_result: NERResult) -> List[EntityStats]:
        """Aggregate entities by (type, normalized_text) with counts and frame info.

        This normalizes unicode variations (e.g., different apostrophes) so that
        "McDonald's" and "McDonald's" are treated as the same entity.
        """
        entity_map: Dict[Tuple[str, str], EntityStats] = {}

        for doc in ner_result.documents:
            for chunk in doc.chunks:
                frame_id = chunk.frame_id
                for entity in chunk.entities:
                    # Use normalized text as the key
                    normalized = normalize_entity_text(entity.text)
                    key = (entity.type, normalized)
                    if key not in entity_map:
                        entity_map[key] = EntityStats(
                            text=entity.text,  # Keep original text for display
                            entity_type=entity.type,
                            count=0,
                            frames={},
                        )
                    stats = entity_map[key]
                    stats.count += 1
                    stats.frames[frame_id] = stats.frames.get(frame_id, 0) + 1

        return list(entity_map.values())

    def _llm_consolidate_all(
        self,
        entity_stats: List[EntityStats],
        domain: str,
    ) -> List[ConsolidatedEntity]:
        """Process all entities through LLM in batches."""
        all_consolidated = []
        entity_id_counter = 1

        # Process in batches
        pbar = tqdm(
            total=len(entity_stats),
            desc=f"Consolidating entities",
            unit="entity",
        )

        for i in range(0, len(entity_stats), self.batch_size):
            batch = entity_stats[i : i + self.batch_size]
            try:
                batch_results = self._llm_consolidate_batch(batch, domain)
                # Assign entity IDs
                for entity in batch_results:
                    entity.entity_id = f"ent_{entity_id_counter:04d}"
                    entity_id_counter += 1
                all_consolidated.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Batch consolidation failed: {e}")
                # Continue with next batch
            finally:
                pbar.update(len(batch))

        pbar.close()
        return all_consolidated

    def _llm_consolidate_batch(
        self,
        batch: List[EntityStats],
        domain: str,
    ) -> List[ConsolidatedEntity]:
        """Send a batch of entities to LLM for consolidation."""
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(batch, domain)

        # Call LLM
        try:
            response = self.llm_client.infer(system_prompt, user_prompt)
            # Parse JSON response
            parsed = self._parse_llm_response(response, batch)
            return parsed
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # Fallback: return unconsolidated entities
            return self._fallback_entities(batch)

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return """You are an entity consolidation expert. Given a list of named entities extracted from news articles, you will:
1. Group entities that refer to the same real-world entity
2. Correct any misclassified entity types
3. Add enrichment information

Entity types: PERSON, ORG (organization), EVENT, FAC (facility), PRODUCT

Output valid JSON only."""

    def _build_user_prompt(self, batch: List[EntityStats], domain: str) -> str:
        """Build user prompt for a batch of entities."""
        entities_text = "\n".join(
            f'- "{e.text}" ({e.entity_type}, count: {e.count})' for e in batch
        )

        guidance_text = f"\nContext: {self.guidance}\n" if self.guidance else ""

        prompt = f"""Consolidate these entities extracted from news articles about {domain}:{guidance_text}
Entities to process:
{entities_text}

Instructions:
1. Group entities that refer to the same real-world entity
2. Correct misclassified types (e.g., "Veganuary" is ORG not PERSON, "Beyond Meat" is ORG not WORK_OF_ART)
3. Add attributes based on entity type
4. DISCARD invalid entities (parsing errors like "I've", partial words, etc.) by omitting them

Valid types: PERSON, ORG, EVENT, FAC, PRODUCT (WORK_OF_ART should become ORG or PRODUCT)

Return JSON array:
[
  {{
    "canonical_name": "preferred name",
    "entity_type": "PERSON|ORG|EVENT|FAC|PRODUCT",
    "aliases": ["all", "name", "variations", "including original"],
    "attributes": {{
      // For PERSON: {{"organization": "affiliated org if known"}}
      // For ORG: {{"organization_type": "company|ngo|govt|media|research|other"}}
    }},
    "confidence": 0.0-1.0
  }}
]

Omit any entity that should be discarded (invalid, parsing error, not a real entity)."""
        return prompt

    def _parse_llm_response(
        self,
        response: str,
        batch: List[EntityStats],
    ) -> List[ConsolidatedEntity]:
        """Parse LLM JSON response into ConsolidatedEntity objects."""
        try:
            # Try to extract JSON from response
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

            # Build lookup for original entities using normalized text
            entity_lookup = {normalize_entity_text(e.text): e for e in batch}

            consolidated = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                entity_type = item.get("entity_type", "ORG")
                canonical_name = item.get("canonical_name", "").strip()
                if not canonical_name:
                    continue

                # Normalize display name (fix all-caps, etc.)
                canonical_name = normalize_display_name(canonical_name, entity_type)

                aliases = item.get("aliases", [])
                if not isinstance(aliases, list):
                    aliases = [canonical_name]

                # Compute total count and frames from aliases
                total_count = 0
                frames: Dict[str, int] = {}
                original_types: Dict[str, int] = {}

                for alias in aliases:
                    normalized_alias = normalize_entity_text(alias)
                    if normalized_alias in entity_lookup:
                        stats = entity_lookup[normalized_alias]
                        total_count += stats.count
                        for frame_id, count in stats.frames.items():
                            frames[frame_id] = frames.get(frame_id, 0) + count
                        original_types[stats.entity_type] = original_types.get(stats.entity_type, 0) + 1

                if total_count == 0:
                    # Try finding by canonical name
                    normalized_canonical = normalize_entity_text(canonical_name)
                    if normalized_canonical in entity_lookup:
                        stats = entity_lookup[normalized_canonical]
                        total_count = stats.count
                        frames = stats.frames.copy()
                        original_types = {stats.entity_type: 1}

                consolidated.append(
                    ConsolidatedEntity(
                        entity_id="",  # Will be assigned later
                        canonical_name=canonical_name,
                        entity_type=item.get("entity_type", "ORG"),
                        aliases=aliases,
                        total_count=total_count,
                        attributes=item.get("attributes", {}),
                        original_types=original_types,
                        frames=frames,
                        confidence=float(item.get("confidence", 1.0)),
                    )
                )

            return consolidated

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_entities(batch)

    def _fallback_entities(self, batch: List[EntityStats]) -> List[ConsolidatedEntity]:
        """Fallback: return unconsolidated entities as-is."""
        return [
            ConsolidatedEntity(
                entity_id="",
                canonical_name=normalize_display_name(e.text, e.entity_type),
                entity_type=e.entity_type,
                aliases=[e.text],
                total_count=e.count,
                attributes={},
                original_types={e.entity_type: 1},
                frames=e.frames.copy(),
                confidence=1.0,
            )
            for e in batch
        ]

    def _merge_duplicate_entities(
        self,
        entities: List[ConsolidatedEntity],
    ) -> List[ConsolidatedEntity]:
        """Merge entities with identical (normalized_name, type).

        Sometimes the LLM creates separate consolidated entities for what should
        be the same entity. For example, "Beyond Meat" might appear twice:
        - Once for instances originally typed as ORG (correct)
        - Once for instances typed as WORK_OF_ART (corrected to ORG)

        This merges them into a single entity with combined counts and metadata.
        """
        # Group by (normalized_name, type)
        entity_map: Dict[Tuple[str, str], ConsolidatedEntity] = {}

        for entity in entities:
            normalized_name = normalize_entity_text(entity.canonical_name)
            key = (normalized_name, entity.entity_type)

            if key not in entity_map:
                entity_map[key] = entity
            else:
                # Merge with existing entity
                existing = entity_map[key]

                # Combine aliases (deduplicated by normalized form)
                existing_aliases_normalized = {normalize_entity_text(a) for a in existing.aliases}
                for alias in entity.aliases:
                    if normalize_entity_text(alias) not in existing_aliases_normalized:
                        existing.aliases.append(alias)

                # Sum counts
                existing.total_count += entity.total_count

                # Merge attributes (keep existing, add new if not present)
                for attr_key, attr_val in entity.attributes.items():
                    if attr_key not in existing.attributes:
                        existing.attributes[attr_key] = attr_val

                # Merge original_types counts
                for orig_type, count in entity.original_types.items():
                    existing.original_types[orig_type] = (
                        existing.original_types.get(orig_type, 0) + count
                    )

                # Merge frame counts
                for frame_id, count in entity.frames.items():
                    existing.frames[frame_id] = existing.frames.get(frame_id, 0) + count

                # Use higher confidence
                existing.confidence = max(existing.confidence, entity.confidence)

        return list(entity_map.values())

    def _build_type_corrections(
        self,
        entity_stats: List[EntityStats],
        consolidated: List[ConsolidatedEntity],
    ) -> Dict[str, Dict[str, int]]:
        """Build type corrections map: original_type -> corrected_type -> count."""
        corrections: Dict[str, Dict[str, int]] = {}

        for entity in consolidated:
            corrected_type = entity.entity_type
            for original_type, count in entity.original_types.items():
                if original_type != corrected_type:
                    if original_type not in corrections:
                        corrections[original_type] = {}
                    corrections[original_type][corrected_type] = (
                        corrections[original_type].get(corrected_type, 0) + count
                    )

        return corrections


__all__ = ["NERConsolidator", "EntityStats"]
