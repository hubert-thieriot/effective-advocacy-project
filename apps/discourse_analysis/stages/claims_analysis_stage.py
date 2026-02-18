"""
Claims Analysis stage for discourse analysis pipeline.

Replaces/augments stance detection with Discourse Network Analysis style:
1. Extract statements from chunks with actors (from NER)
2. Induce/load claims (specific policy positions)
3. Score agreement between statements and claims
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

from efi_core.pipeline import PipelineStage
from efi_analyser.claims import (
    StatementExtractor,
    ClaimInducer,
    AgreementScorer,
    Statement,
    ClaimSchema,
    StatementsResult,
    AgreementsResult,
    ClaimsAnalysisResult,
)
from efi_analyser.scorers.openai_interface import OpenAIInterface, OpenAIConfig

from .base import StageContext


class ClaimsAnalysisStage(PipelineStage[StageContext, ClaimsAnalysisResult]):
    """Extract statements, induce/load claims, score agreement.

    This stage implements Discourse Network Analysis methodology where:
    - Statements: What actors say in text chunks
    - Claims: Specific policy positions (NOT abstract themes)
    - Agreements: How statements align with claims
    """

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        return bool(input_data.config.claims_analysis.enabled)

    def execute(self, input_data: StageContext) -> ClaimsAnalysisResult:
        """Execute claims analysis with three substages."""
        config = input_data.config
        paths = input_data.paths

        # Substage 1: Extract statements from NER chunks
        statements = self._extract_statements(input_data)

        if not statements.statements:
            self.logger.warning("No statements extracted, returning empty result")
            return ClaimsAnalysisResult(
                statements=statements,
                claims=ClaimSchema(domain=config.claims_analysis.domain or ""),
                agreements=AgreementsResult(),
            )

        # Substage 2: Get or create claim schema
        claims = self._get_or_create_claims(input_data, statements.statements)

        if not claims.claims:
            self.logger.warning("No claims available, returning empty agreements")
            return ClaimsAnalysisResult(
                statements=statements,
                claims=claims,
                agreements=AgreementsResult(),
            )

        # Substage 3: Score agreements
        agreements = self._score_agreements(input_data, statements.statements, claims)

        return ClaimsAnalysisResult(
            statements=statements,
            claims=claims,
            agreements=agreements,
        )

    def _extract_statements(self, input_data: StageContext) -> StatementsResult:
        """Substage 1: Extract statements from NER chunks."""
        from efi_analyser.ner import NERResult

        config = input_data.config.claims_analysis
        paths = input_data.paths
        state = input_data.state

        statements_path = paths.statements_path

        # Check for cached results
        if statements_path.exists() and (config.reload_statements or input_data.config.regenerate_report_only):
            try:
                cached = StatementsResult.load(statements_path)
                self.logger.info(f"Loaded cached statements: {cached.n_statements} statements")
                return cached
            except Exception as e:
                self.logger.warning(f"Failed to load cached statements: {e}")

        # Need NER result with entities - try state first, then disk
        ner_result = state.ner_result
        if not ner_result and paths.ner_entities_path.exists():
            try:
                ner_result = NERResult.load(paths.ner_entities_path)
                self.logger.info(f"Loaded NER result from disk: {ner_result.n_entities} entities")
            except Exception as e:
                self.logger.warning(f"Failed to load NER from disk: {e}")

        if not ner_result:
            self.logger.warning("No NER result available for statement extraction")
            return StatementsResult(statements=[])

        # Check if there are entities
        if ner_result.n_entities == 0:
            self.logger.warning("NER result has no entities")
            return StatementsResult(statements=[])

        self.logger.info(
            f"Extracting statements from {ner_result.n_chunks} chunks "
            f"with {ner_result.n_entities} entities"
        )

        # Initialize LLM client
        stmt_config = config.statement
        llm_config = OpenAIConfig(
            model=stmt_config.model,
            temperature=stmt_config.temperature,
            flex_processing=stmt_config.flex_processing,
        )
        llm_client = OpenAIInterface(name="statement_extraction", config=llm_config)

        # Initialize extractor
        extractor = StatementExtractor(
            llm_client=llm_client,
            batch_size=stmt_config.batch_size,
            max_statements_per_chunk=stmt_config.max_statements_per_chunk,
            min_statement_length=stmt_config.min_statement_length,
        )

        # Extract statements
        result = extractor.extract(
            ner_result,
            consolidated=ner_result.consolidated,
            show_progress=True,
            chunk_limit=stmt_config.chunk_limit,
        )

        # Save results
        result.save(statements_path)
        self.logger.info(f"Extracted {result.n_statements} statements")

        return result

    def _get_or_create_claims(
        self,
        input_data: StageContext,
        statements: List[Statement],
    ) -> ClaimSchema:
        """Substage 2: Get claims from file or induce from statements."""
        config = input_data.config.claims_analysis
        paths = input_data.paths
        claims_config = config.claims

        claims_path = paths.claims_schema_path

        # Option 1: Load from file
        if claims_config.source == "file":
            if not claims_config.schema_path:
                raise ValueError(
                    "claims_analysis.claims.schema_path is required when source='file'"
                )
            self.logger.info(f"Loading claims from file: {claims_config.schema_path}")
            return ClaimSchema.load(claims_config.schema_path)

        # Option 2: Induction
        # Check for cached results
        if claims_path.exists() and (config.reload_claims or input_data.config.regenerate_report_only):
            try:
                cached = ClaimSchema.load(claims_path)
                self.logger.info(f"Loaded cached claims: {len(cached.claims)} claims")
                return cached
            except Exception as e:
                self.logger.warning(f"Failed to load cached claims: {e}")

        # Sample statements for induction
        sample_size = min(claims_config.induction_size, len(statements))
        if sample_size < len(statements):
            sampled = random.sample(statements, sample_size)
        else:
            sampled = list(statements)

        self.logger.info(
            f"Inducing claims from {len(sampled)} statements (of {len(statements)} total)"
        )

        # Initialize LLM client
        llm_config = OpenAIConfig(
            model=claims_config.induction_model,
            temperature=claims_config.induction_temperature,
            flex_processing=claims_config.flex_processing,
        )
        llm_client = OpenAIInterface(name="claim_induction", config=llm_config)

        # Initialize inducer
        domain = config.domain or input_data.config.framing.domain or ""
        inducer = ClaimInducer(
            llm_client=llm_client,
            domain=domain,
            claim_target=claims_config.claim_target,
            induction_guidance=claims_config.induction_guidance,
            max_statements_per_call=claims_config.induction_batch_size,
        )

        # Induce claims
        result = inducer.induce(sampled)

        # Save results
        result.save(claims_path)
        self.logger.info(f"Induced {len(result.claims)} claims")

        return result

    def _score_agreements(
        self,
        input_data: StageContext,
        statements: List[Statement],
        claims: ClaimSchema,
    ) -> AgreementsResult:
        """Substage 3: Score statement-claim agreements."""
        config = input_data.config.claims_analysis
        paths = input_data.paths
        scoring_config = config.scoring

        agreements_path = paths.agreements_path

        # Check for cached results
        if agreements_path.exists() and not config.reload_agreements:
            if input_data.config.regenerate_report_only or not input_data.allow_new_work:
                try:
                    cached = AgreementsResult.load(agreements_path)
                    self.logger.info(
                        f"Loaded cached agreements: {cached.n_agreements} agreements"
                    )
                    return cached
                except Exception as e:
                    self.logger.warning(f"Failed to load cached agreements: {e}")

        self.logger.info(
            f"Scoring {len(statements)} statements against {len(claims.claims)} claims"
        )

        # Initialize LLM client
        llm_config = OpenAIConfig(
            model=scoring_config.model,
            temperature=scoring_config.temperature,
            flex_processing=scoring_config.flex_processing,
        )
        llm_client = OpenAIInterface(name="agreement_scoring", config=llm_config)

        # Initialize scorer
        scorer = AgreementScorer(
            llm_client=llm_client,
            support_threshold=scoring_config.support_threshold,
            oppose_threshold=scoring_config.oppose_threshold,
            batch_size=scoring_config.batch_size,
        )

        # Score agreements
        result = scorer.score(statements, claims, show_progress=True)

        # Save results
        result.save(agreements_path)
        self.logger.info(f"Scored {result.n_agreements} statement-claim pairs")

        return result

    def get_metadata(self) -> dict:
        return {"stage": self.name}


__all__ = ["ClaimsAnalysisStage"]
