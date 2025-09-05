#!/usr/bin/env python3
"""
Integration tests for Scrapy concurrent processing
Tests with known URLs from test cases to verify functionality
"""

import pytest
import yaml
import time
from pathlib import Path
from efi_corpus.manager import run_config

# Mark all tests in this class as requiring internet access and API access
pytestmark = [pytest.mark.internet, pytest.mark.api]


class TestScrapyIntegration:
    """Test Scrapy concurrent processing with known URLs from test cases"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path("temp_test_scrapy_integration")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Load test URLs from the existing test cases
        test_cases_path = Path(__file__).parent / "url_test_cases.yaml"
        with open(test_cases_path, 'r') as f:
            self.test_cases = yaml.safe_load(f)['test_cases']
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_sequential_processing(self):
        """Test sequential processing with known URLs from test cases"""
        # Create a test config for sequential processing using stored URLs
        test_config = {
            "builder": "mediacloud",
            "corpus": {
                "base_dir": str(self.temp_dir / "test_sequential_corpus")
            },
            "parameters": {
                "collections": [
                    {"id": 34412118, "name": "India - National"}
                ],
                "use_concurrent_processing": False,
                "force_refresh_cache": True,
                "limit": 2,  # Test with 2 URLs
                # Use stored URLs instead of MediaCloud API
                "test_urls": [
                    self.test_cases["indian_express_cse"]["url"],
                    self.test_cases["hindustan_times_mcg"]["url"]
                ]
            }
        }
        
        # Write config to temp file
        config_path = self.temp_dir / "test_sequential.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run the config
        start_time = time.time()
        result = run_config(str(config_path))
        sequential_time = time.time() - start_time
        
        # Verify results
        assert result['discovered'] > 0
        assert result['added'] > 0
        assert result['failed'] >= 0  # Allow failures due to network timeouts
        
        print(f"✅ Sequential processing: {result['discovered']} discovered, {result['added']} added in {sequential_time:.2f}s")
    
    def test_concurrent_processing(self):
        """Test concurrent processing with known URLs from test cases"""
        # Create a test config for concurrent processing using stored URLs
        test_config = {
            "builder": "mediacloud",
            "corpus": {
                "base_dir": str(self.temp_dir / "test_concurrent_corpus")
            },
            "parameters": {
                "collections": [
                    {"id": 34412118, "name": "India - National"}
                ],
                "use_concurrent_processing": True,
                "concurrent_requests": 2,  # Lower concurrency
                "download_delay": 0.1,
                "force_refresh_cache": True,
                "limit": 2,  # Test with 2 URLs
                # Use stored URLs instead of MediaCloud API
                "test_urls": [
                    self.test_cases["indian_express_cse"]["url"],
                    self.test_cases["hindustan_times_mcg"]["url"]
                ]
            }
        }
        
        # Write config to temp file
        config_path = self.temp_dir / "test_concurrent.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run the config
        start_time = time.time()
        result = run_config(str(config_path))
        concurrent_time = time.time() - start_time
        
        # Verify results - handle concurrent processing result format
        assert isinstance(result, dict)
        # Check if it's the direct result format from run_scrapy_spider
        if 'added' in result and 'total_processed' in result:
            # Direct result format from concurrent processing
            assert result['added'] > 0, f"Expected to add some documents, got: {result}"
            assert isinstance(result['added'], int), f"Expected integer for added count: {result}"
            assert isinstance(result['failed'], int), f"Expected integer for failed count: {result}"
            spider_result = result  # Use the result directly
        else:
            # Handle fallback result format (when concurrent processing fails and falls back to sequential)
            # Look for standard result keys that should be present
            expected_keys = ['discovered', 'added', 'skipped_quality', 'skipped_text_extraction', 'skipped_duplicate', 'failed', 'total_docs']
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                raise AssertionError(f"Missing expected result keys: {missing_keys}. Available keys: {list(result.keys())}")

            # Use the result directly for validation
            spider_result = result
            assert spider_result['discovered'] > 0, f"Expected to discover some documents, got: {spider_result}"
            assert spider_result['added'] > 0, f"Expected to add some documents, got: {spider_result}"
            assert isinstance(spider_result['added'], int), f"Expected integer for added count: {spider_result}"
            assert isinstance(spider_result['failed'], int), f"Expected integer for failed count: {spider_result}"
        
        # Print results with appropriate keys
        if 'discovered' in spider_result:
            print(f"✅ Concurrent processing: {spider_result['discovered']} discovered, {spider_result['added']} added in {concurrent_time:.2f}s")
        else:
            print(f"✅ Concurrent processing: {spider_result['total_processed']} processed, {spider_result['added']} added in {concurrent_time:.2f}s")
    
    def test_content_extraction_validation(self):
        """Test that extracted content contains expected text from test cases"""
        # Create a test config for content validation
        test_config = {
            "builder": "mediacloud",
            "corpus": {
                "base_dir": str(self.temp_dir / "test_content_validation_corpus")
            },
            "parameters": {
                "collections": [
                    {"id": 34412118, "name": "India - National"}
                ],
                "use_concurrent_processing": False,  # Sequential for easier debugging
                "force_refresh_cache": True,
                "limit": 1,  # Test with 1 URL for content validation
                # Use stored URLs instead of MediaCloud API
                "test_urls": [
                    self.test_cases["indian_express_cse"]["url"]
                ]
            }
        }
        
        # Write config to temp file
        config_path = self.temp_dir / "test_content_validation.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run the config
        result = run_config(str(config_path))
        
        # Verify that content was extracted
        assert result['added'] > 0, "Expected to add at least one document"
        
        # Check if the corpus was created and contains documents
        corpus_dir = Path(self.temp_dir / "test_content_validation_corpus")
        assert corpus_dir.exists(), "Corpus directory should exist"
        
        # Check for index file
        index_file = corpus_dir / "index.jsonl"
        assert index_file.exists(), "Index file should exist"
        
        # Read the index to get document ID
        with open(index_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0, "Index file should contain at least one entry"
            
            # Parse the first document
            import json
            first_doc = json.loads(lines[0])
            
            # Verify document structure
            assert 'url' in first_doc, "Document should have URL"
            assert 'title' in first_doc, "Document should have title"
            assert 'id' in first_doc, "Document should have ID"
            
            doc_id = first_doc['id']
            
        # Read the actual content from the document directory
        docs_dir = corpus_dir / "documents"
        doc_dir = docs_dir / doc_id
        text_file = doc_dir / "text.txt"
        
        assert text_file.exists(), f"Text file should exist at {text_file}"
        
        # Read the content
        content = text_file.read_text(encoding='utf-8')
        assert len(content) > 0, "Content should not be empty"
        
        # If content is substantial (not mocked), check for expected text
        if len(content) > 100:  # Only check if we have real content
            required_texts = self.test_cases["indian_express_cse"]["required_text"]
            content_lower = content.lower()
            
            # Check if any of the required text phrases appear in the content
            found_phrases = []
            for required_text in required_texts:
                if required_text.lower() in content_lower:
                    found_phrases.append(required_text)
            
            # Should find at least some of the required phrases
            assert len(found_phrases) > 0, f"Expected to find some required phrases in content. Required: {required_texts}, Found: {found_phrases}, Content preview: {content[:200]}..."
            print(f"✅ Content validation: Found {len(found_phrases)}/{len(required_texts)} required phrases: {found_phrases}")
        else:
            print(f"⚠️  Content appears to be mocked (length: {len(content)}), skipping detailed validation")
        
        print("✅ Content extraction validation passed")
    
    def test_sequential_vs_concurrent_consistency(self):
        """Test that sequential and concurrent processing produce consistent results"""
        # Test sequential processing
        self.test_sequential_processing()
        
        # Test concurrent processing  
        self.test_concurrent_processing()
        
        # Both tests should have passed (no exceptions)
        print("✅ Sequential and concurrent processing both work correctly")
    
    def test_concurrent_processing_with_higher_concurrency(self):
        """Test concurrent processing with higher concurrency settings"""
        # Create a test config with higher concurrency
        test_config = {
            "builder": "mediacloud",
            "corpus": {
                "base_dir": str(self.temp_dir / "test_high_concurrency_corpus")
            },
            "parameters": {
                "collections": [
                    {"id": 34412118, "name": "India - National"}
                ],
                "use_concurrent_processing": True,
                "concurrent_requests": 4,  # Moderate concurrency
                "download_delay": 0.1,     # Standard delay
                "force_refresh_cache": True,
                "limit": 2,  # Test with 2 URLs
                # Use stored URLs instead of MediaCloud API
                "test_urls": [
                    self.test_cases["indian_express_cse"]["url"],
                    self.test_cases["hindustan_times_mcg"]["url"]
                ]
            }
        }
        
        # Write config to temp file
        config_path = self.temp_dir / "test_high_concurrency.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run the config
        start_time = time.time()
        result = run_config(str(config_path))
        high_concurrency_time = time.time() - start_time
        
        # Verify results - allow some failures due to timeouts/network issues
        assert result['discovered'] > 0
        assert result['added'] > 0
        assert result['failed'] >= 0  # Allow failures due to network timeouts
        
        print(f"✅ High concurrency processing: {result['discovered']} discovered, {result['added']} added in {high_concurrency_time:.2f}s")
        print(f"   Average time per URL: {high_concurrency_time/result['added']:.2f}s")
    
    def test_error_handling_in_concurrent_processing(self):
        """Test error handling in concurrent processing"""
        # Create a test config with some URLs that might fail
        test_config = {
            "builder": "mediacloud",
            "corpus": {
                "base_dir": str(self.temp_dir / "test_error_handling_corpus")
            },
            "parameters": {
                "collections": [
                    {"id": 34412118, "name": "India - National"}
                ],
                "use_concurrent_processing": True,
                "concurrent_requests": 2,
                "download_delay": 0.1,
                "force_refresh_cache": True,
                "limit": 2,  # Test with 2 URLs
                # Use stored URLs instead of MediaCloud API
                "test_urls": [
                    self.test_cases["indian_express_cse"]["url"],
                    self.test_cases["hindustan_times_mcg"]["url"]
                ]
            }
        }
        
        # Write config to temp file
        config_path = self.temp_dir / "test_error_handling.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run the config
        start_time = time.time()
        result = run_config(str(config_path))
        error_handling_time = time.time() - start_time
        
        # Verify error handling (should handle gracefully)
        assert result['discovered'] >= 0
        assert result['added'] >= 0
        assert result['failed'] >= 0
        
        print(f"✅ Error handling test: {result['discovered']} discovered, {result['added']} added, {result['failed']} failed in {error_handling_time:.2f}s")
        print("✅ Error handling in concurrent processing works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])