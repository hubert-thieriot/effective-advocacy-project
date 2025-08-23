"""
Simple Example: Keyword Presence Analysis
"""

from pathlib import Path
from efi_analyser import WordOccurrenceApp

corpus_path = Path("corpora/air_quality/India")
app = WordOccurrenceApp(corpus_path,
                        keywords=["air pollution", "coal", "asthma", "transport", "health",
                                  "transport", "power", "industry", "biomass"])
result = app.run()

if result.data:
    data = result.data.aggregated_data
    print(f"Keyword presence analysis:")
    print(f"Total documents: {data['total_documents']}")
    print()
    for keyword, count in data['keyword_counts'].items():
        percentage = data['keyword_percentages'][keyword]
        print(f"{keyword}: {count} docs ({percentage:.1f}%)")
else:
    print("No results")
