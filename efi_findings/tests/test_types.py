"""
Tests for the types module
"""

import pytest
from datetime import datetime
from ..types import Finding, DocumentFindings, ExtractionConfig, StorageConfig
from pathlib import Path


class TestFinding:
    """Test the Finding dataclass"""
    
    def test_finding_creation(self):
        """Test creating a Finding with minimal data"""
        finding = Finding(text="Test finding")
        assert finding.text == "Test finding"
        assert finding.confidence is None
        assert finding.category is None
        assert finding.keywords is None
    
    def test_finding_with_all_fields(self):
        """Test creating a Finding with all fields"""
        finding = Finding(
            text="Test finding with all fields",
            confidence=0.95,
            category="policy",
            keywords=["test", "policy", "finding"]
        )
        assert finding.text == "Test finding with all fields"
        assert finding.confidence == 0.95
        assert finding.category == "policy"
        assert finding.keywords == ["test", "policy", "finding"]


class TestDocumentFindings:
    """Test the DocumentFindings dataclass"""
    
    def test_document_findings_creation(self):
        """Test creating DocumentFindings with minimal data"""
        findings = DocumentFindings(url="https://example.com")
        assert findings.url == "https://example.com"
        assert findings.title is None
        assert findings.published_at is None
        assert findings.language is None
        assert findings.findings == []
        assert findings.metadata == {}
        assert isinstance(findings.extraction_date, datetime)
    
    def test_document_findings_with_all_fields(self):
        """Test creating DocumentFindings with all fields"""
        findings = DocumentFindings(
            url="https://example.com",
            title="Test Document",
            published_at=datetime(2023, 1, 1),
            language="en",
            findings=[Finding(text="Test finding")],
            metadata={"source": "test"}
        )
        assert findings.url == "https://example.com"
        assert findings.title == "Test Document"
        assert findings.published_at == datetime(2023, 1, 1)
        assert findings.language == "en"
        assert len(findings.findings) == 1
        assert findings.findings[0].text == "Test finding"
        assert findings.metadata["source"] == "test"


class TestExtractionConfig:
    """Test the ExtractionConfig dataclass"""
    
    def test_extraction_config_defaults(self):
        """Test ExtractionConfig default values"""
        config = ExtractionConfig()
        assert config.model == "gpt-3.5-turbo"
        assert config.max_tokens == 1000
        assert config.temperature == 0.1
        assert config.system_prompt is None
        assert config.extraction_prompt is None
    
    def test_extraction_config_custom(self):
        """Test ExtractionConfig with custom values"""
        config = ExtractionConfig(
            model="gpt-4",
            max_tokens=2000,
            temperature=0.2,
            system_prompt="Custom system prompt",
            extraction_prompt="Custom extraction prompt"
        )
        assert config.model == "gpt-4"
        assert config.max_tokens == 2000
        assert config.temperature == 0.2
        assert config.system_prompt == "Custom system prompt"
        assert config.extraction_prompt == "Custom extraction prompt"


class TestStorageConfig:
    """Test the StorageConfig dataclass"""
    
    def test_storage_config_defaults(self):
        """Test StorageConfig default values"""
        config = StorageConfig()
        assert config.storage_path == Path("findings")
        assert config.index_filename == "findings_index.jsonl"
        assert config.findings_dir == "findings"
        assert config.backup_enabled is True
    
    def test_storage_config_custom(self):
        """Test StorageConfig with custom values"""
        config = StorageConfig(
            storage_path=Path("custom_findings"),
            index_filename="custom_index.jsonl",
            findings_dir="custom_findings",
            backup_enabled=False
        )
        assert config.storage_path == Path("custom_findings")
        assert config.index_filename == "custom_index.jsonl"
        assert config.findings_dir == "custom_findings"
        assert config.backup_enabled is False
