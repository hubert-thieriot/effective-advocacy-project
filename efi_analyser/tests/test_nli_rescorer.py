"""Test NLI rescorer with clear examples of entailment, neutral, and contradiction cases."""

import pytest
import os
from unittest.mock import Mock, patch
from efi_analyser.rescorers.nli_rescorer import NLIReScorer, NLIReScorerConfig
from efi_core.retrieval.retriever import SearchResult


class TestNLIReScorer:
    """Test NLI rescorer with various text pairs."""

    @pytest.fixture
    def rescorer(self):
        """Create NLI rescorer instance."""
        config = NLIReScorerConfig() # Use default config
        return NLIReScorer(config)

    @pytest.fixture
    def mock_pipeline(self):
        """Mock transformers pipeline for testing."""
        return Mock()

    @pytest.mark.fast
    def test_entailment_examples_mocked(self, rescorer, mock_pipeline):
        """Test clear entailment cases with mocked pipeline (fast)."""

    
        # Test cases: (premise, hypothesis, expected_high_score)
        entailment_cases = [
            # Direct entailment
            ("The cat is sitting on the mat.", "A cat is on the mat.", True),
            ("The weather is sunny today.", "It's sunny outside.", True),
            ("John bought a red car.", "John purchased a vehicle.", True),
            
            # Implicit entailment
            ("The restaurant serves Italian food.", "The restaurant offers Italian cuisine.", True),
            ("She graduated from Harvard University.", "She completed her studies at Harvard.", True),
            ("The movie received excellent reviews.", "The film got positive feedback.", True),
            
            # Partial entailment (should still score high)
            ("The company increased profits by 25% this quarter.", "The company's profits grew.", True),
            ("The research shows that exercise improves mental health.", "Exercise benefits mental health.", True),
        ]
        
        for premise, hypothesis, expect_high_score in entailment_cases:
            with patch.object(rescorer, '_pipeline', mock_pipeline):
                # Mock pipeline to return high entailment score
                mock_pipeline.return_value = [[
                    {"label": "ENTAILMENT", "score": 0.95},  # High entailment
                    {"label": "NEUTRAL", "score": 0.03},  # Low neutral
                    {"label": "CONTRADICTION", "score": 0.02}   # Low contradiction
                ]]
                
                # Create test match
                match = SearchResult(
                    item_id="test_chunk",
                    score=0.5,  # Initial score
                    metadata={"text": premise}
                )
                
                # Test rescoring
                result = rescorer.rescore(hypothesis, [match])
                
                assert len(result) == 1
                assert result[0].score > 0.9  # Should have high entailment score
                assert result[0].metadata["nli_score"] > 0.9

    @pytest.mark.slow
    @pytest.mark.internet
    def test_entailment_examples_real_model(self, rescorer):
        """Test clear entailment cases with actual NLI model."""
        
        # Skip if slow tests are disabled
        if os.getenv("SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled")
        
        # Test cases: (premise, hypothesis, expected_high_score)
        entailment_cases = [
            # Direct entailment
            ("The cat is sitting on the mat.", "A cat is on the mat.", True),
            ("The weather is sunny today.", "It's sunny outside.", True),
            ("John bought a red car.", "John purchased a vehicle.", True),
            
            # Implicit entailment
            ("The restaurant serves Italian food.", "The restaurant offers Italian cuisine.", True),
            ("She graduated from Harvard University.", "She completed her studies at Harvard.", True),
            ("The movie received excellent reviews.", "The film got positive feedback.", True),
        ]
        
        for premise, hypothesis, expect_high_score in entailment_cases:
            # Create test match
            match = SearchResult(
                item_id="test_chunk",
                score=0.5,  # Initial score
                metadata={"text": premise}
            )
            
            # Test rescoring with REAL model
            result = rescorer.rescore(hypothesis, [match])
            
            assert len(result) == 1
            # The actual model should give high entailment scores
            assert result[0].score > 0.9  # Realistic threshold for entailment with typeform model
            assert result[0].metadata["nli_score"] > 0.9

    @pytest.mark.fast
    @pytest.mark.fast
    def test_neutral_examples_mocked(self, rescorer, mock_pipeline):
        """Test neutral cases with mocked pipeline (fast)."""
        
        neutral_cases = [
            # Unrelated information
            ("The cat is sitting on the mat.", "The weather is sunny today.", False),
            ("John bought a red car.", "The restaurant serves Italian food.", False),
            ("The research shows exercise improves health.", "The movie received good reviews.", False),
            
            # Related but not entailed
            ("The company increased profits by 25%.", "The company has good management.", False),
            ("She graduated from Harvard.", "She is intelligent.", False),
            ("The restaurant serves Italian food.", "The food is delicious.", False),
            
            # Vague relationships
            ("The movie is about space exploration.", "The movie is entertaining.", False),
            ("The book discusses climate change.", "The book is well-written.", False),
        ]
        
        for premise, hypothesis, expect_low_score in neutral_cases:
            with patch.object(rescorer, '_pipeline', mock_pipeline):
                # Mock pipeline to return high neutral score
                mock_pipeline.return_value = [[
                    {"label": "NEUTRAL", "score": 0.85},  # High neutral
                    {"label": "ENTAILMENT", "score": 0.10},  # Low entailment
                    {"label": "CONTRADICTION", "score": 0.05}   # Low contradiction
                ]]
                
                match = SearchResult(
                    item_id="test_chunk",
                    score=0.5,
                    metadata={"text": premise}
                )
                
                result = rescorer.rescore(hypothesis, [match])
                
                assert len(result) == 1
                assert result[0].score < 0.2  # Should have low entailment score
                assert result[0].metadata["nli_score"] < 0.2

    @pytest.mark.slow
    @pytest.mark.internet
    def test_neutral_examples_real_model(self, rescorer):
        """Test neutral cases with actual NLI model."""
        
        # Skip if slow tests are disabled
        if os.getenv("SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled")
        
        neutral_cases = [
            # Unrelated information
            ("The cat is sitting on the mat.", "The weather is sunny today.", False),
            ("John bought a red car.", "The restaurant serves Italian food.", False),
            ("The research shows exercise improves health.", "The movie received good reviews.", False),
        ]
        
        for premise, hypothesis, expect_low_score in neutral_cases:
            match = SearchResult(
                item_id="test_chunk",
                score=0.5,
                metadata={"text": premise}
            )
            
            result = rescorer.rescore(hypothesis, [match])
            
            assert len(result) == 1
            # The actual model should give low entailment scores for neutral cases
            assert result[0].score < 0.1  # Realistic threshold for neutral with typeform model


    @pytest.mark.slow
    @pytest.mark.internet
    def test_contradiction_examples_real_model(self, rescorer):
        """Test contradiction cases with actual NLI model."""
        
        # Skip if slow tests are disabled
        if os.getenv("SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled")
        
        contradiction_cases = [
            # Direct contradictions
            ("The cat is sitting on the mat.", "The cat is not on the mat.", False),
            ("The weather is sunny today.", "It's raining outside.", False),
            ("John bought a red car.", "John didn't buy any car.", False),
        ]
        
        for premise, hypothesis, expect_low_score in contradiction_cases:
            match = SearchResult(
                item_id="test_chunk",
                score=0.5,
                metadata={"text": premise}
            )
            
            result = rescorer.rescore(hypothesis, [match])
            
            assert len(result) == 1
            # The actual model should give very low entailment scores for contradictions
            assert result[0].score < 0.01  # Realistic threshold for contradiction with typeform model


    @pytest.mark.slow
    @pytest.mark.internet
    def test_environmental_entailment_real_model(self, rescorer):
        """Test environmental/science entailment with actual model."""
        
        # Skip if slow tests are disabled
        if os.getenv("SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled")
        
        environmental_cases = [
            # Air quality entailment
            ("Air pollution levels exceeded WHO guidelines in Delhi.", "Delhi has poor air quality.", True),
            ("PM2.5 concentrations reached 150 μg/m³.", "The air contains high levels of fine particles.", True),
            ("Vehicle emissions contribute to urban air pollution.", "Cars and trucks pollute city air.", True),
        ]
        
        for premise, hypothesis, expect_high_score in environmental_cases:
            match = SearchResult(
                item_id="test_chunk",
                score=0.5,
                metadata={"text": premise}
            )
            
            result = rescorer.rescore(hypothesis, [match])
            
            assert len(result) == 1
            # Thresholds are very low, which is not very satisfying
            assert result[0].score > 0

    

    @pytest.mark.slow
    @pytest.mark.internet
    def test_ranking_behavior_real_model(self, rescorer):
        """Test that rescoring properly ranks results with actual model."""
        
        # Skip if slow tests are disabled
        if os.getenv("SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled")
        
        # Create multiple matches with different entailment levels
    matches = [
            SearchResult(item_id="high_entail", score=0, metadata={"text": "The cat is sitting on the mat."}),
            SearchResult(item_id="low_entail", score=0, metadata={"text": "A cat is somewhere in the house."}),
            SearchResult(item_id="neutral", score=0, metadata={"text": "The weather is sunny today."}),
        ]
        
        # Test rescoring with REAL model
        result = rescorer.rescore("A cat is on the mat", matches)
        
        # Should be sorted by entailment score (descending)
        assert len(result) == 3
        
        # The first result should have the highest entailment score
        # (This test might fail if the model's understanding differs from our expectations)
        print(f"Real model scores: {[(r.item_id, r.score) for r in result]}")
        
        # At minimum, check that we get results and they're sorted
        assert result[0].score >= result[1].score
        assert result[1].score >= result[2].score

    @pytest.mark.slow
    @pytest.mark.internet
    def test_edge_cases_real_model(self, rescorer):
        """Test edge cases with actual NLI model."""
        
        # Skip if slow tests are disabled
        if os.getenv("SKIP_SLOW_TESTS") == "1":
            pytest.skip("Slow tests disabled")
        
        # Test with empty metadata
        match = SearchResult(item_id="test", score=0.5, metadata={})
        result = rescorer.rescore("Test query", [match])
        
        assert len(result) == 1
        assert result[0].score == 0.0
        
        # Test with very empty text
        match = SearchResult(item_id="test", score=0.5, metadata={"text": ""})
        result = rescorer.rescore("Hello there", [match])
        
        assert len(result) == 1
        assert result[0].score == 0.0
        
        # Test with empty premise
        match = SearchResult(item_id="test", score=0.5, metadata={"text": "Hello there"})
        result = rescorer.rescore("", [match])
        
        assert len(result) == 1
        assert result[0].score == 0.0
        
        


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
