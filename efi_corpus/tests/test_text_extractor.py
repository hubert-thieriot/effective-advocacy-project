"""
Tests for the text extractor and content fetching pipeline
"""

import pytest
from pathlib import Path
import sys
import yaml

from efi_corpus.text_extractor import TextExtractor

def load_test_cases():
    """Load test cases from YAML configuration file"""
    config_path = Path(__file__).parent / "url_test_cases.yaml"
    
    if not config_path.exists():
        print(f"Warning: Test cases config not found at {config_path}")
        return []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('test_cases', {}).items()
    except Exception as e:
        print(f"Error loading test cases: {e}")
        return []


class TestContentFetchingAndExtraction:
    """Test cases for the complete content fetching and text extraction pipeline"""
    
    @pytest.fixture(scope="class")
    def test_cases(self):
        """Load test cases from configuration"""
        return load_test_cases()
    
    def test_urls_fetch_and_extract_required_text(self, test_cases):
        """Test that URLs can be fetched and their content contains required text"""
        import requests
        
        if not test_cases:
            pytest.skip("No test cases loaded")
        
        extractor = TextExtractor()
        
        for test_name, test_case in test_cases:
            url = test_case['url']
            required_texts = test_case['required_text']
            description = test_case['description']
            expected_min_length = test_case.get('expected_min_length', 1000)
            site = test_case.get('site', 'unknown')
            
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Verify we got HTML content
                assert response.content, f"No content fetched from {url}"
                assert len(response.content) > 1000, f"Fetched content too small: {len(response.content)} bytes"
                
                result = extractor.extract_text(response.content, 'html', url)
                
                # Verify text was extracted
                assert result.get('text'), f"No text was extracted from {url}"
                text = result.get('text', '')
                
                # Verify text length meets minimum requirement
                assert len(text) >= expected_min_length, f"Text too short from {url}: {len(text)} chars (expected >= {expected_min_length})"
                
                # Verify each required text is present
                missing_texts = []
                for required_text in required_texts:
                    if required_text not in text:
                        missing_texts.append(required_text)
                
                if missing_texts:
                    pytest.fail(f"Missing required text from {url}: {missing_texts}")
                
                print(f"✅ {site}: {len(text)} chars, all required text found")
                
            except Exception as e:
                pytest.fail(f"Failed to process {url}: {e}")


