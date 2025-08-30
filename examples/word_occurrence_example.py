#!/usr/bin/env python3
"""
Word Occurrence Example

Clean example of using the WordOccurrenceApp to analyze keyword
occurrences in a corpus.
"""

from pathlib import Path
from efi_analyser.apps import WordOccurrenceApp
from efi_analyser.types import WordOccurrenceConfig


def main():
    """Run word occurrence analysis."""
    
    # Configuration
    config = WordOccurrenceConfig(
        corpus_path=Path("test_corpus"),
        workspace_path=Path("workspace"),
        keywords=[
            "air quality",
            "pollution",
            "PM2.5",
            "PM10",
            "AQI",
            "emissions",
            "clean air",
            "environmental health"
        ],
        case_sensitive=False,
        whole_word_only=True,
        allow_hyphenation=True,
        output_formats=["csv", "json", "html"]
    )
    
    # Initialize app
    app = WordOccurrenceApp(config)
    
    # Run analysis
    results = app.analyze_keywords()
    
    # Generate reports
    report_paths = app.generate_reports(results)
    
    # Output summary
    print(f"Analyzed {results.total_documents} documents")
    print(f"Keywords: {', '.join(results.keywords)}")
    print(f"Reports saved to: {config.workspace_path}")
    
    # Show keyword statistics
    print("\nKeyword Statistics:")
    for keyword in results.keywords:
        count = results.keyword_counts.get(keyword, 0)
        percentage = results.keyword_percentages.get(keyword, 0.0)
        print(f"  {keyword}: {count} documents ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
