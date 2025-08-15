#!/usr/bin/env python3
"""
Extract key findings from 10 CREA publications
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from efi_findings import (
    LibraryBuilder,
    CREAPublicationsCollector,
    FindingExtractorFromText,
    FindingExtractorFromUrl,
    JSONStorer,
    ExtractionConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_crea_findings(max_publications=20):
    """Extract findings from CREA publications"""
    
    print("🔍 CREA Publications Findings Extraction")
    print("=" * 60)
    print(f"Target: {max_publications} publications")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Configure extraction
    extraction_config = ExtractionConfig(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.1
    )
    
    # Step 2: Create components
    print("📋 Setting up components...")
    
    # Collector: Find CREA publication URLs
    collector = CREAPublicationsCollector(max_sources=max_publications)
    
    # Extractors: Text extraction first, URL extraction as fallback
    text_extractor = FindingExtractorFromText(extraction_config)
    url_extractor = FindingExtractorFromUrl(extraction_config)
    
    # Storer: Save findings to organized structure
    storer = JSONStorer("crea_publications")
    
    print(f"   ✅ Collector: {collector.__class__.__name__}")
    print(f"   ✅ Extractors: {[e.__class__.__name__ for e in [text_extractor, url_extractor]]}")
    print(f"   ✅ Storer: {storer.__class__.__name__}")
    print()
    
    # Step 3: Build the findings library
    print("🚀 Building findings library...")
    
    with LibraryBuilder(
        name="crea_publications",
        collector=collector,
        extractors=[text_extractor, url_extractor],
        storer=storer,
        extraction_config=extraction_config
    ) as builder:
        
        # Build the library
        summary = builder.build_library(max_sources=max_publications)
        
        # Display comprehensive results
        print("\n📊 EXTRACTION RESULTS")
        print("=" * 60)
        print(f"📚 Library Name: {summary['library_name']}")
        print(f"⏰ Build Time: {summary['build_timestamp']}")
        print(f"🔗 Total Sources: {summary['total_sources_processed']}")
        print(f"✅ Successful: {summary['successful_extractions']}")
        print(f"❌ Failed: {summary['failed_extractions']}")
        print(f"📈 Success Rate: {summary['success_rate']}%")
        print(f"💡 Total Findings: {summary['total_findings_extracted']}")
        print(f"📊 Avg Findings/Source: {summary['average_findings_per_source']}")
        
        # Show detailed findings
        print("\n📖 DETAILED FINDINGS")
        print("=" * 60)
        
        all_findings = storer.list_all_findings()
        
        for i, doc_findings in enumerate(all_findings, 1):
            print(f"\n📄 Publication {i}: {doc_findings.title or 'No title'}")
            print(f"   🔗 URL: {doc_findings.url}")
            print(f"   📅 Extraction Date: {doc_findings.extraction_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   💡 Findings: {len(doc_findings.findings)}")
            
            # Show all findings for this publication
            for j, finding in enumerate(doc_findings.findings, 1):
                print(f"      {j}. {finding.text}")
            
            print(f"   {'─' * 80}")
        
        # Show search examples
        print("\n🔍 SEARCH EXAMPLES")
        print("=" * 60)
        
        search_queries = [
            "China", "India", "emissions", "steel", "air quality", 
            "renewable", "fossil fuel", "policy", "target"
        ]
        
        for query in search_queries:
            results = builder.search_library(query)
            if results:
                print(f"\n🔎 '{query}': {len(results)} documents found")
                
                for doc_findings in results:
                    matching_count = sum(1 for f in doc_findings.findings 
                                       if query.lower() in f.text.lower())
                    print(f"   📄 {doc_findings.title or 'No title'}")
                    print(f"      {matching_count} findings mention '{query}'")
        
        # Show cache and storage statistics
        print("\n💾 SYSTEM STATISTICS")
        print("=" * 60)
        
        # Cache stats
        print("📦 Cache Information:")
        for extractor in [text_extractor, url_extractor]:
            cache_stats = extractor.get_cache_stats()
            print(f"   {extractor.name} extractor:")
            print(f"      Cached items: {cache_stats.get('cached_items', 0)}")
            print(f"      Cache size: {cache_stats.get('cache_size_mb', 0)} MB")
        
        # Storage stats
        print("\n💿 Storage Information:")
        storage_stats = storer.get_storage_stats()
        print(f"   Total documents: {storage_stats.get('total_documents', 0)}")
        print(f"   Total findings: {storage_stats.get('total_findings', 0)}")
        print(f"   File size: {storage_stats.get('file_size_mb', 0)} MB")
        print(f"   Storage path: {storage_stats.get('storage_path', 'Unknown')}")
        
        # Export findings
        print("\n📤 EXPORTING FINDINGS")
        print("=" * 60)
        
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        export_path = results_dir / f"crea_findings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        exported_path = builder.export_library(export_path)
        
        print(f"✅ Findings exported to: {exported_path}")
        
        # Create a summary report
        summary_report = {
            "extraction_summary": summary,
            "publications_processed": len(all_findings),
            "total_findings": sum(len(doc.findings) for doc in all_findings),
            "extraction_date": datetime.now().isoformat(),
            "cache_stats": {
                "text_extractor": text_extractor.get_cache_stats(),
                "url_extractor": url_extractor.get_cache_stats()
            },
            "storage_stats": storage_stats
        }
        
        summary_file = results_dir / f"crea_extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Summary report saved to: {summary_file}")
        
        print("\n" + "=" * 60)
        print("🎉 CREA Findings Extraction Completed Successfully!")
        print(f"📁 Check the 'libraries/crea_publications' directory for stored findings")
        print(f"📁 Check the 'cache/finding_extraction_*' directories for cached extractions")
        print(f"📁 Check the 'results/' directory for exported files")
        
        return summary, all_findings

def main():
    """Main function to run the CREA findings extraction"""
    try:
        summary, findings = extract_crea_findings(max_publications=None)
        
        # Print final summary
        print(f"\n🎯 FINAL SUMMARY")
        print(f"   Successfully processed {summary['total_sources_processed']} CREA publications")
        print(f"   Extracted {summary['total_findings_extracted']} key findings")
        print(f"   Success rate: {summary['success_rate']}%")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
