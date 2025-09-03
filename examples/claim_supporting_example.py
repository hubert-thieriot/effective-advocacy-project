#!/usr/bin/env python3
"""
Claim Supporting Example

Clean example of using the ClaimSupportingApp to analyze claims against
a corpus using NLI scoring and classification.
"""

from pathlib import Path
from efi_analyser.apps import ClaimSupportingApp
from efi_analyser.types import ClaimSupportingConfig


def main():
    """Run claim supporting analysis."""
    
    # Configuration
    config = ClaimSupportingConfig(
        corpus_path=Path("test_corpus"),
        workspace_path=Path("workspace"),
        cosine_threshold=0.1,
        top_k_retrieval=1000,
        nli_model="typeform/distilbert-base-uncased-mnli",
        classification_threshold=0.7,
        output_formats=["html"]
    )
    
    # Initialize app
    app = ClaimSupportingApp(config)
    
    # Define claims to analyze
    claims = [
        "Air quality in India has improved significantly in recent years",
        "PM2.5 levels in Indian cities exceed WHO guidelines",
        "Industrial emissions are the primary cause of air pollution in India",
        "The government has implemented effective air quality monitoring systems"
    ]
    
    # Run analysis
    results = app.analyze_claims(claims)
    
    # Generate reports
    report_paths = app.generate_reports(results)
    
    # Output summary
    print(f"Analyzed {len(claims)} claims")
    print(f"Total chunks processed: {sum(r.total_chunks for r in results)}")
    print(f"Reports saved to: {config.workspace_path}")


if __name__ == "__main__":
    main()
