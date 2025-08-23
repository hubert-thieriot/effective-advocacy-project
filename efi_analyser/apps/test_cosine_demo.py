"""
Test script for the Cosine Similarity Demo App

This script creates sample data and runs the demo to verify it works correctly.
"""

import tempfile
import json
from pathlib import Path

from cosine_similarity_demo import CosineSimilarityDemo


def create_sample_library(library_path: Path):
    """Create a sample CREA library for testing"""
    library_path.mkdir(parents=True, exist_ok=True)
    
    # Sample findings data
    findings_data = [
        {
            "url": "https://crea.org/research1",
            "title": "CREA Research on Air Quality",
            "published_at": "2025-01-01T00:00:00",
            "language": "en",
            "extraction_date": "2025-01-20T00:00:00",
            "findings": [
                {
                    "text": "Air pollution from coal power plants causes serious health problems in urban areas.",
                    "confidence": 0.9,
                    "category": "health",
                    "keywords": ["air pollution", "coal", "health", "urban"]
                },
                {
                    "text": "Renewable energy sources significantly reduce carbon emissions and improve air quality.",
                    "confidence": 0.95,
                    "category": "environment",
                    "keywords": ["renewable energy", "carbon", "emissions", "air quality"]
                },
                {
                    "text": "Industrial emissions contribute to climate change and respiratory diseases.",
                    "confidence": 0.88,
                    "category": "climate",
                    "keywords": ["industrial", "emissions", "climate change", "respiratory"]
                }
            ],
            "metadata": {"source": "CREA research"}
        },
        {
            "url": "https://crea.org/research2",
            "title": "CREA Analysis of Energy Sources",
            "published_at": "2025-01-02T00:00:00",
            "language": "en",
            "extraction_date": "2025-01-20T00:00:00",
            "findings": [
                {
                    "text": "Solar and wind energy are cost-effective alternatives to fossil fuels.",
                    "confidence": 0.92,
                    "category": "energy",
                    "keywords": ["solar", "wind", "cost-effective", "fossil fuels"]
                },
                {
                    "text": "Electric vehicles reduce air pollution in cities and dependence on oil.",
                    "confidence": 0.87,
                    "category": "transport",
                    "keywords": ["electric vehicles", "air pollution", "cities", "oil"]
                }
            ],
            "metadata": {"source": "CREA analysis"}
        }
    ]
    
    # Save findings
    findings_path = library_path / "findings.json"
    with open(findings_path, 'w') as f:
        json.dump(findings_data, f, indent=2)
    
    # Save metadata
    metadata = {
        "name": "CREA Research Library",
        "description": "Sample CREA research findings on air quality and energy",
        "created_at": "2025-01-20T00:00:00"
    }
    with open(library_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created sample library at: {library_path}")
    return library_path


def create_sample_corpus(corpus_path: Path):
    """Create a sample air quality corpus (India) for testing"""
    corpus_path.mkdir(parents=True, exist_ok=True)
    
    # Sample documents
    documents = [
        {
            "doc_id": "india_air_quality_1",
            "title": "Air Quality in Delhi: A Critical Analysis",
            "text": "Delhi faces severe air quality challenges due to industrial emissions, vehicular pollution, and agricultural burning. The city's air quality index frequently exceeds hazardous levels, leading to respiratory problems and reduced life expectancy. Coal-fired power plants in the region contribute significantly to particulate matter pollution.",
            "source": "Environmental Research Journal",
            "metadata": {"region": "Delhi", "focus": "air quality"}
        },
        {
            "doc_id": "india_air_quality_2", 
            "title": "Renewable Energy Solutions for Indian Cities",
            "text": "Indian cities are increasingly adopting renewable energy solutions to combat air pollution. Solar power installations have grown rapidly, providing clean electricity and reducing dependence on coal. Wind energy projects in coastal regions are also contributing to cleaner air and reduced carbon emissions.",
            "source": "Energy Policy Review",
            "metadata": {"region": "India", "focus": "renewable energy"}
        },
        {
            "doc_id": "india_air_quality_3",
            "title": "Health Impacts of Air Pollution in Mumbai",
            "text": "Mumbai's air quality has deteriorated due to industrial activities and traffic congestion. Studies show increased cases of asthma, bronchitis, and cardiovascular diseases among residents. The city is implementing electric vehicle incentives and public transportation improvements to address these issues.",
            "source": "Public Health Research",
            "metadata": {"region": "Mumbai", "focus": "health impacts"}
        },
        {
            "doc_id": "india_air_quality_4",
            "title": "Agricultural Burning and Air Quality in Punjab",
            "text": "Agricultural burning in Punjab contributes to seasonal air quality degradation across northern India. The practice releases large amounts of particulate matter and greenhouse gases. Alternative farming methods and waste management solutions are being explored to reduce this environmental impact.",
            "source": "Agricultural Studies",
            "metadata": {"region": "Punjab", "focus": "agricultural practices"}
        }
    ]
    
    # Save documents
    for doc in documents:
        # Create document directory structure
        doc_dir = corpus_path / "documents" / doc["doc_id"]
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Save text file
        text_path = doc_dir / "text.txt"
        with open(text_path, 'w') as f:
            f.write(doc['text'])
        
        # Save metadata file
        meta_path = doc_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump({
                "title": doc["title"],
                "source": doc["source"],
                "metadata": doc["metadata"]
            }, f, indent=2)
        
        # Save fetch info file
        fetch_path = doc_dir / "fetch.json"
        with open(fetch_path, 'w') as f:
            json.dump({
                "url": f"https://example.com/{doc['doc_id']}",
                "fetched_at": "2025-01-20T00:00:00",
                "status": "success"
            }, f, indent=2)
    
    # Create manifest
    manifest = {
        "corpus_name": "India Air Quality Corpus",
        "description": "Sample corpus of air quality research from India",
        "created_at": "2025-01-20T00:00:00",
        "documents": [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "metadata": doc["metadata"]
            }
            for doc in documents
        ]
    }
    
    manifest_path = corpus_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create index.jsonl file that CorpusHandle expects
    index_path = corpus_path / "index.jsonl"
    with open(index_path, 'w') as f:
        for doc in documents:
            index_entry = {
                "id": doc["doc_id"],
                "title": doc["title"],
                "source": doc["source"],
                "metadata": doc["metadata"],
                "file_path": f"{doc['doc_id']}.txt"
            }
            f.write(json.dumps(index_entry) + '\n')
    
    print(f"Created sample corpus at: {corpus_path}")
    return corpus_path


def main():
    """Main test function"""
    print("Creating sample data for Cosine Similarity Demo...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample data
        library_path = temp_path / "crea_library"
        corpus_path = temp_path / "india_air_quality_corpus"
        workspace_path = temp_path / "workspace"
        
        create_sample_library(library_path)
        create_sample_corpus(corpus_path)
        
        print("\nRunning Cosine Similarity Demo...")
        print("="*50)
        
        try:
            # Run the demo
            demo = CosineSimilarityDemo(library_path, corpus_path, workspace_path)
            results = demo.run_demo()
            
            print("\n" + "="*50)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"Total findings processed: {results['total_findings']}")
            print(f"Total documents processed: {results['total_documents']}")
            print(f"Total comparisons: {results['total_comparisons']}")
            print(f"Duration: {results['duration_seconds']:.2f} seconds")
            print(f"Library embeddings cached: {results['cache_stats']['library_embeddings_cached']}")
            print(f"Corpus embeddings cached: {results['cache_stats']['corpus_embeddings_cached']}")
            
            # Show top result
            if results['findings_results']:
                top_result = results['findings_results'][0]
                print(f"\nTop finding: {top_result['finding']['text'][:80]}...")
                print(f"Average similarity: {top_result['avg_similarity']:.3f}")
                if top_result['top_matches']:
                    top_match = top_result['top_matches'][0]
                    print(f"Best document match: {top_match['doc_title']}")
                    print(f"Similarity score: {top_match['similarity']:.3f}")
            
            return 0
            
        except Exception as e:
            print(f"Error running demo: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    exit(main())
