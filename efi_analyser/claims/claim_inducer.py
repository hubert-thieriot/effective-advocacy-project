"""Claim induction for discovering policy positions from statements."""

from __future__ import annotations

import json
import logging
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import Claim, ClaimSchema, Statement


class ClaimInducer:
    """Generate a claim schema from statements using an LLM backend.

    Claims are specific policy positions/beliefs, NOT abstract themes.

    Good claims:
    - "DNA testing should be mandatory for racehorses"
    - "Factory farming should be banned"
    - "We should adopt cap and trade for carbon emissions"

    Bad claims (too abstract):
    - "Animal welfare" (not a position)
    - "Environmental impact" (not actionable)
    """

    MAX_STATEMENTS_PER_CALL: int = 400
    MAX_TOTAL_STATEMENTS: int = 1000
    MAX_CHARS_PER_STATEMENT: int = 500

    SYSTEM_PROMPT_TEMPLATE: str = """You are an expert at identifying policy positions and claims in discourse analysis.

DOMAIN: {domain}
TARGET: {claim_target}

INSTRUCTIONS:
Identify the specific policy positions that actors are debating in the statements provided.

1. Claims must be SPECIFIC and ACTIONABLE positions — not abstract themes
2. Each claim should be debatable (actors can support or oppose it)
3. Claims must be relevant to the domain
4. Avoid overlapping claims — merge similar positions
5. Extract supporting and opposing examples from the statements

Good claims: "Factory farming should be banned", "Carbon emissions should be taxed"
Bad claims: "Animal welfare" (not a position), "Environmental concerns" (not actionable)
{guidance_section}
OUTPUT FORMAT:
Return valid JSON only:
{{
  "domain": "{domain}",
  "claims": [
    {{
      "claim_id": "claim_01",
      "text": "the specific policy position",
      "description": "clarification of what this claim means",
      "keywords": ["related", "terms"],
      "examples": ["statement supporting this claim..."],
      "counter_examples": ["statement opposing this claim..."]
    }}
  ],
  "notes": "optional notes about the claim set"
}}"""

    def __init__(
        self,
        llm_client: Any,
        domain: str,
        claim_target: int | str = 10,
        induction_guidance: Optional[str] = None,
        max_chars_per_statement: Optional[int] = None,
        max_statements_per_call: Optional[int] = None,
        max_total_statements: Optional[int] = None,
        infer_timeout: Optional[float] = 600,
    ) -> None:
        """Initialize the inducer.

        Args:
            llm_client: LLM interface for API calls
            domain: Domain/topic context (e.g., "Grand National horse racing debate")
            claim_target: Target number of claims (int or string like "8-12")
            induction_guidance: Optional guidance/seed claims for induction
            max_chars_per_statement: Max characters per statement (truncates)
            max_statements_per_call: Max statements per LLM call
            max_total_statements: Max total statements to process
            infer_timeout: Timeout for LLM calls
        """
        self.llm_client = llm_client
        self.domain = domain
        self.claim_target = claim_target
        self.induction_guidance = induction_guidance.strip() if induction_guidance else ""
        self.infer_timeout = infer_timeout
        self.logger = logging.getLogger(self.__class__.__name__)

        self.max_chars_per_statement = max_chars_per_statement or self.MAX_CHARS_PER_STATEMENT
        self.max_statements_per_call = max_statements_per_call or self.MAX_STATEMENTS_PER_CALL
        self.max_total_statements = max_total_statements or self.MAX_TOTAL_STATEMENTS

        self._claim_target_text = self._format_claim_target(claim_target)

    def induce(self, statements: Iterable[Statement]) -> ClaimSchema:
        """Induce claims from a collection of statements.

        Args:
            statements: Statements extracted from text chunks

        Returns:
            ClaimSchema with induced claims
        """
        # Prepare statements (dedupe, truncate)
        prepared = self._prepare_statements(statements)
        if not prepared:
            raise ValueError("Claim induction requires at least one statement.")

        self.logger.info(f"Inducing claims from {len(prepared)} statements")

        if len(prepared) <= self.max_statements_per_call:
            return self._induce_single(prepared)

        # For large sets: batch then merge
        partial_schemas: List[ClaimSchema] = []
        for chunk in tqdm(self._chunk_statements(prepared)):
            partial_schemas.append(self._induce_single(chunk))

        return self._merge_schemas(prepared, partial_schemas)

    def _prepare_statements(self, statements: Iterable[Statement]) -> List[Statement]:
        """Prepare statements for induction (dedupe, truncate)."""
        seen_texts = set()
        unique: List[Statement] = []

        for stmt in statements:
            text = stmt.text.strip()
            if not text or text in seen_texts:
                continue

            # Truncate if needed
            if len(text) > self.max_chars_per_statement:
                text = text[: self.max_chars_per_statement].rstrip() + "..."
                stmt = Statement(
                    statement_id=stmt.statement_id,
                    chunk_id=stmt.chunk_id,
                    doc_id=stmt.doc_id,
                    text=text,
                    speaker=stmt.speaker,
                    speaker_entity_id=stmt.speaker_entity_id,
                    context=stmt.context,
                )

            seen_texts.add(stmt.text)
            unique.append(stmt)

            if len(unique) >= self.max_total_statements:
                break

        return unique

    def _chunk_statements(self, statements: Sequence[Statement]) -> List[Sequence[Statement]]:
        """Split statements into chunks for batch processing."""
        chunks = []
        for i in range(0, len(statements), self.max_statements_per_call):
            chunks.append(statements[i : i + self.max_statements_per_call])
        return chunks

    def _build_system_prompt(self) -> str:
        """Build system prompt with domain, target, and guidance."""
        guidance_section = ""
        if self.induction_guidance:
            guidance_section = (
                f"\nGUIDANCE:\n{self.induction_guidance}\n"
                "Use these as starting points but refine, split, or merge "
                "as needed based on the statements.\n"
            )
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            domain=self.domain,
            claim_target=self._claim_target_text,
            guidance_section=guidance_section,
        )

    def _induce_single(self, statements: Sequence[Statement]) -> ClaimSchema:
        """Induce claims from a single batch of statements."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(statements)

        try:
            response = self.llm_client.infer(system_prompt, user_prompt)
            return self._parse_response(response)
        except Exception as e:
            self.logger.error(f"Claim induction failed: {e}")
            raise

    def _build_user_prompt(self, statements: Sequence[Statement]) -> str:
        """Build user prompt with statements as input."""
        stmt_lines = []
        for i, stmt in enumerate(statements, 1):
            speaker_part = f" [{stmt.speaker}]" if stmt.speaker else ""
            stmt_lines.append(f"{i}.{speaker_part} {stmt.text}")

        return "STATEMENTS:\n" + "\n".join(stmt_lines)

    def _merge_schemas(
        self,
        all_statements: Sequence[Statement],
        partial_schemas: Sequence[ClaimSchema],
    ) -> ClaimSchema:
        """Merge multiple partial claim schemas into one."""
        if not partial_schemas:
            raise ValueError("No partial schemas to merge")
        if len(partial_schemas) == 1:
            return partial_schemas[0]

        # Build summary of partial claims
        claims_summary = []
        for batch_idx, schema in enumerate(partial_schemas, 1):
            for claim in schema.claims:
                claims_summary.append({
                    "batch": batch_idx,
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "description": claim.description[:200],
                    "keywords": claim.keywords[:5],
                })

        claims_json = json.dumps(claims_summary, ensure_ascii=False, indent=2)

        # Sample some statements for context
        sample_limit = min(30, len(all_statements))
        sample_text = "\n".join(
            f"- {s.text}" for s in all_statements[:sample_limit]
        )

        user_prompt = f"""Consolidate these partial claims from multiple batches into a unified set.

PARTIAL CLAIMS FROM BATCHES:
{claims_json}

SAMPLE STATEMENTS (for context):
{sample_text}"""

        response = self.llm_client.infer(self._build_system_prompt(), user_prompt)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> ClaimSchema:
        """Parse LLM JSON response into ClaimSchema."""
        content = response.strip()

        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            snippet = content[:200].replace("\n", " ")
            raise ValueError(f"Claim inducer returned invalid JSON: {e}: {snippet}") from e

        if not isinstance(payload, dict):
            raise ValueError("Claim inducer expected JSON object at top level")

        domain = payload.get("domain", self.domain)
        claims_payload = payload.get("claims", [])
        notes = payload.get("notes", "")

        if not isinstance(claims_payload, list):
            raise ValueError("Claim inducer response 'claims' must be a list")

        if not claims_payload:
            return ClaimSchema(domain=str(domain), claims=[], notes=str(notes))
        
        claims: List[Claim] = []
        for entry in claims_payload:
            if not isinstance(entry, dict):
                continue

            claim_id = entry.get("claim_id")
            text = entry.get("text", "").strip()

            if not claim_id or not text:
                continue

            claims.append(
                Claim(
                    claim_id=str(claim_id),
                    text=text,
                    description=str(entry.get("description", "")),
                    keywords=[str(k) for k in entry.get("keywords", []) if isinstance(k, str)],
                    examples=[str(e) for e in entry.get("examples", []) if isinstance(e, str)],
                    counter_examples=[str(e) for e in entry.get("counter_examples", []) if isinstance(e, str)],
                    source="induced",
                )
            )

        if not claims:
            raise ValueError("Claim inducer did not return any valid claims")

        return ClaimSchema(domain=str(domain), claims=claims, notes=str(notes))

    def _format_claim_target(self, target: int | str) -> str:
        """Format claim target as string."""
        if isinstance(target, str):
            return target.strip()
        if target is None:
            return ""
        return f"{int(target)} claims"


__all__ = ["ClaimInducer"]
