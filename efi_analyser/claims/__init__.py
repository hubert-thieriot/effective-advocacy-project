"""Claims analysis module for discourse network analysis.

This module provides tools for:
- Extracting statements from text chunks with identified speakers
- Inducing claims (specific policy positions) from statements
- Scoring agreement between statements and claims
"""

from .types import (
    Statement,
    Claim,
    ClaimSchema,
    Agreement,
    StatementsResult,
    AgreementsResult,
    ClaimsAnalysisResult,
)
from .statement_extractor import StatementExtractor
from .claim_inducer import ClaimInducer
from .agreement_scorer import AgreementScorer

__all__ = [
    # Types
    "Statement",
    "Claim",
    "ClaimSchema",
    "Agreement",
    "StatementsResult",
    "AgreementsResult",
    "ClaimsAnalysisResult",
    # Core classes
    "StatementExtractor",
    "ClaimInducer",
    "AgreementScorer",
]
