"""Score statement-claim agreements using LLM.

Key efficiency:
- Claims are in the instructions (cached by OpenAI across calls).
- Multiple statements are batched per LLM call.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from efi_analyser.scorers.openai_interface import OpenAIInterface
from .types import Statement, Claim, ClaimSchema, Agreement, AgreementsResult


class AgreementScorer:
    """Score agreement between statements and claims.

    Claims go into the instructions (cached by OpenAI prompt caching).
    Statements are batched in the input to reduce total API calls.
    """

    SYSTEM_PROMPT_TEMPLATE: str = """You are an expert at analyzing stance and agreement in discourse.

You will be given one or more statements. For each statement, determine how it aligns with each of the policy claims listed below.

For each claim, rate:
- score: -1.0 (strongly opposes) to 1.0 (strongly supports), 0.0 for neutral
- label: "supports", "opposes", "neutral", or "irrelevant"
- rationale: Brief explanation (1-2 sentences)

Guidelines:
- "supports": Statement explicitly or implicitly endorses the claim
- "opposes": Statement explicitly or implicitly argues against the claim
- "neutral": Statement mentions related topic but takes no clear stance
- "irrelevant": Statement has no relation to the claim

Be precise. Only mark as "supports" or "opposes" if there's clear evidence.

CLAIMS TO EVALUATE:
{claims_context}

Output valid JSON only."""

    def __init__(
        self,
        llm_client: OpenAIInterface,
        support_threshold: float = 0.3,
        oppose_threshold: float = -0.3,
        batch_size: int = 5,
    ) -> None:
        """Initialize agreement scorer.

        Args:
            llm_client: OpenAI interface for LLM calls
            support_threshold: Score >= this is labeled "supports"
            oppose_threshold: Score <= this is labeled "opposes"
            batch_size: Number of statements per LLM call
        """
        self.llm_client = llm_client
        self.support_threshold = support_threshold
        self.oppose_threshold = oppose_threshold
        self.batch_size = max(1, batch_size)
        self.logger = logging.getLogger(self.__class__.__name__)

    def score(
        self,
        statements: List[Statement],
        claims: ClaimSchema,
        show_progress: bool = True,
    ) -> AgreementsResult:
        """Score agreement between all statements and all claims.

        Args:
            statements: Statements to score
            claims: Claim schema with claims to score against
            show_progress: Whether to show progress bar

        Returns:
            AgreementsResult with all agreements
        """
        if not statements:
            return AgreementsResult(
                agreements=[],
                scoring_model=self.llm_client.spec_key(),
            )

        if not claims.claims:
            self.logger.warning("No claims provided for scoring")
            return AgreementsResult(
                agreements=[],
                scoring_model=self.llm_client.spec_key(),
            )

        self.logger.info(
            f"Scoring {len(statements)} statements against {len(claims.claims)} claims "
            f"(batch_size={self.batch_size})"
        )

        # Build instructions once — claims are baked in so OpenAI caches them
        instructions = self._build_instructions(claims)

        all_agreements: List[Agreement] = []
        batches = list(self._chunk_statements(statements))

        pbar = tqdm(
            batches,
            desc=f"Scoring agreements ({self.llm_client.spec_key()})",
            unit="batch",
            disable=not show_progress,
        )

        for batch in pbar:
            pbar.set_postfix(stmts=len(batch))
            try:
                agreements = self._score_batch(batch, claims, instructions)
                all_agreements.extend(agreements)
            except Exception as e:
                self.logger.error(f"Failed to score batch: {e}")
                # Fallback: score individually
                for stmt in batch:
                    try:
                        agreements = self._score_batch([stmt], claims, instructions)
                        all_agreements.extend(agreements)
                    except Exception as e2:
                        self.logger.error(
                            f"Failed to score statement {stmt.statement_id}: {e2}"
                        )

        self.logger.info(f"Scored {len(all_agreements)} statement-claim pairs")

        return AgreementsResult(
            agreements=all_agreements,
            scoring_model=self.llm_client.spec_key(),
        )

    def _chunk_statements(
        self, statements: Sequence[Statement]
    ) -> List[Sequence[Statement]]:
        """Split statements into batches."""
        chunks = []
        for i in range(0, len(statements), self.batch_size):
            chunks.append(statements[i : i + self.batch_size])
        return chunks

    def _build_instructions(self, claims: ClaimSchema) -> str:
        """Build instructions with claims baked in (cached by OpenAI)."""
        claims_context = self._build_claims_context(claims)
        return self.SYSTEM_PROMPT_TEMPLATE.format(claims_context=claims_context)

    def _build_claims_context(self, claims: ClaimSchema) -> str:
        """Build context string describing all claims."""
        lines = []
        for claim in claims.claims:
            keywords_str = ", ".join(claim.keywords[:5]) if claim.keywords else ""
            lines.append(
                f"- {claim.claim_id}: \"{claim.text}\"\n"
                f"  Description: {claim.description or 'N/A'}\n"
                f"  Keywords: {keywords_str or 'N/A'}"
            )
        return "\n".join(lines)

    def _score_batch(
        self,
        statements: Sequence[Statement],
        claims: ClaimSchema,
        instructions: str,
    ) -> List[Agreement]:
        """Score a batch of statements against all claims in one LLM call."""
        user_prompt = self._build_user_prompt(statements, claims)

        try:
            response = self.llm_client.infer(instructions, user_prompt)
            return self._parse_response(response, statements, claims)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return [
                Agreement(
                    statement_id=stmt.statement_id,
                    claim_id=claim.claim_id,
                    score=0.0,
                    label="irrelevant",
                    rationale="Scoring failed",
                    confidence=0.0,
                )
                for stmt in statements
                for claim in claims.claims
            ]

    def _build_user_prompt(
        self,
        statements: Sequence[Statement],
        claims: ClaimSchema,
    ) -> str:
        """Build user prompt with one or more statements."""
        claim_ids = [c.claim_id for c in claims.claims]

        if len(statements) == 1:
            # Single statement — flat response format
            stmt = statements[0]
            speaker_part = f" (Speaker: {stmt.speaker})" if stmt.speaker else ""
            return f"""STATEMENT{speaker_part}:
"{stmt.text}"

CONTEXT:
{stmt.context[:300] if stmt.context else "N/A"}

Return JSON mapping claim_id to score:
{{
  "{claim_ids[0] if claim_ids else 'claim_01'}": {{
    "score": 0.0,
    "label": "supports|opposes|neutral|irrelevant",
    "rationale": "Brief explanation..."
  }}
}}

Include ALL claims: {', '.join(claim_ids)}"""

        # Multiple statements — keyed by statement_id
        stmt_blocks = []
        for i, stmt in enumerate(statements, 1):
            speaker_part = f" [{stmt.speaker}]" if stmt.speaker else ""
            context = stmt.context[:200] if stmt.context else "N/A"
            stmt_blocks.append(
                f"{i}. [{stmt.statement_id}]{speaker_part}: \"{stmt.text}\"\n"
                f"   Context: {context}"
            )

        statements_text = "\n\n".join(stmt_blocks)
        stmt_ids = [s.statement_id for s in statements]

        return f"""STATEMENTS:
{statements_text}

For EACH statement, return scores for ALL claims.

Return JSON mapping statement_id to claim scores:
{{
  "{stmt_ids[0]}": {{
    "{claim_ids[0] if claim_ids else 'claim_01'}": {{
      "score": 0.0,
      "label": "supports|opposes|neutral|irrelevant",
      "rationale": "Brief explanation..."
    }}
  }}
}}

Include ALL statements: {', '.join(stmt_ids)}
Include ALL claims: {', '.join(claim_ids)}"""

    def _parse_response(
        self,
        response: str,
        statements: Sequence[Statement],
        claims: ClaimSchema,
    ) -> List[Agreement]:
        """Parse LLM response into Agreement objects."""
        try:
            content = self._clean_json(response)
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("Response must be a JSON object")

            if len(statements) == 1:
                return self._parse_single(data, statements[0], claims)
            return self._parse_batch(data, statements, claims)

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse response: {e}")
            return [
                Agreement(
                    statement_id=stmt.statement_id,
                    claim_id=claim.claim_id,
                    score=0.0,
                    label="irrelevant",
                    rationale="Parse error",
                    confidence=0.0,
                )
                for stmt in statements
                for claim in claims.claims
            ]

    def _parse_single(
        self,
        data: Dict[str, Any],
        statement: Statement,
        claims: ClaimSchema,
    ) -> List[Agreement]:
        """Parse response for a single statement (flat claim_id -> score map)."""
        agreements: List[Agreement] = []
        for claim in claims.claims:
            claim_data = data.get(claim.claim_id, {})
            score, label, rationale = self._extract_score(claim_data)
            agreements.append(
                Agreement(
                    statement_id=statement.statement_id,
                    claim_id=claim.claim_id,
                    score=score,
                    label=label,
                    rationale=rationale,
                    confidence=1.0,
                )
            )
        return agreements

    def _parse_batch(
        self,
        data: Dict[str, Any],
        statements: Sequence[Statement],
        claims: ClaimSchema,
    ) -> List[Agreement]:
        """Parse response for a batch of statements (statement_id -> claim_id -> score)."""
        agreements: List[Agreement] = []
        for stmt in statements:
            stmt_data = data.get(stmt.statement_id, {})
            if not isinstance(stmt_data, dict):
                self.logger.warning(
                    f"Missing or invalid data for statement {stmt.statement_id}"
                )
                stmt_data = {}
            for claim in claims.claims:
                claim_data = stmt_data.get(claim.claim_id, {})
                score, label, rationale = self._extract_score(claim_data)
                agreements.append(
                    Agreement(
                        statement_id=stmt.statement_id,
                        claim_id=claim.claim_id,
                        score=score,
                        label=label,
                        rationale=rationale,
                        confidence=1.0 if stmt_data else 0.0,
                    )
                )
        return agreements

    def _extract_score(
        self, claim_data: Any
    ) -> tuple[float, str, str]:
        """Extract score, label, rationale from claim data."""
        if isinstance(claim_data, dict):
            score = float(claim_data.get("score", 0.0))
            label = claim_data.get("label", "")
            rationale = claim_data.get("rationale", "")
        else:
            score = float(claim_data) if claim_data else 0.0
            label = ""
            rationale = ""

        score = max(-1.0, min(1.0, score))
        if not label:
            label = self._score_to_label(score)
        return score, label, rationale

    def _clean_json(self, response: str) -> str:
        """Strip markdown code fences from response."""
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to label."""
        if score >= self.support_threshold:
            return "supports"
        elif score <= self.oppose_threshold:
            return "opposes"
        elif abs(score) < 0.1:
            return "irrelevant"
        else:
            return "neutral"


__all__ = ["AgreementScorer"]
