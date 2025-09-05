"""
TATA Stance Detection Scorer

Implementation of the TATA (Topic-Agnostic and Topic-Aware) stance detection model
as a scorer for the EFI corpus system.

Based on: https://github.com/hanshanley/tata
Paper: TATA: Stance Detection via Topic-Agnostic and Topic-Aware Embeddings (EMNLP 2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
from pydantic import BaseModel

from .stance_scorer import StanceScorer


class StanceTATAScorerConfig(BaseModel):
    """Configuration for TATA stance detection scorer."""
    
    model_path: str = "models/tata.pt"
    tokenizer_name: str = "microsoft/deberta-base"
    batch_size: int = 8
    max_length: int = 512
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: bool = False


class TATAModel(nn.Module):
    """
    TATA Model Architecture
    
    This implementation loads the actual TATA model weights and uses a single
    DeBERTa encoder for feature extraction (simplified approach).
    """
    
    def __init__(self, state_dict, config):
        super().__init__()
        self.config = config
        
        # Load a single DeBERTa encoder for feature extraction
        # (In the full TATA model, there are separate TAG and TAW encoders)
        self.encoder = AutoModel.from_pretrained("microsoft/deberta-base")
        
        # Attention mechanism to combine TAG and TAW
        self.attention_encoding = nn.Linear(768, 768)
        
        # Linear layers for processing
        self.linear_tag = nn.Linear(768, 768)
        self.linear_taw = nn.Linear(768, 768)
        self.linear_11 = nn.Linear(1536, 768)  # 768 + 768 = 1536
        self.linear_2 = nn.Linear(768, 768)
        self.linear_21 = nn.Linear(1536, 768)
        self.linear_3 = nn.Linear(768, 3)  # 3 stance classes
        
        # Batch normalization
        self.batchnorm_2 = nn.BatchNorm1d(768)
        
        # Final classifier
        self.total_linear = nn.Linear(1536, 768)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Load the weights from state dict
        self._load_weights(state_dict)
        
    def _load_weights(self, state_dict):
        """Load weights from the state dict, excluding the encoder weights."""
        # Load only the custom layers, not the DeBERTa encoders
        custom_layers = [
            'attention_encoding', 'linear_tag', 'linear_taw', 'linear_11',
            'linear_2', 'linear_21', 'linear_3', 'batchnorm_2', 'total_linear'
        ]
        
        for layer_name in custom_layers:
            weight_key = f"{layer_name}.weight"
            bias_key = f"{layer_name}.bias"
            
            if weight_key in state_dict:
                getattr(self, layer_name).weight.data = state_dict[weight_key]
            if bias_key in state_dict:
                getattr(self, layer_name).bias.data = state_dict[bias_key]
        
        # Handle batch norm running stats
        if 'batchnorm_2.running_mean' in state_dict:
            self.batchnorm_2.running_mean = state_dict['batchnorm_2.running_mean']
        if 'batchnorm_2.running_var' in state_dict:
            self.batchnorm_2.running_var = state_dict['batchnorm_2.running_var']
        if 'batchnorm_2.num_batches_tracked' in state_dict:
            self.batchnorm_2.num_batches_tracked = state_dict['batchnorm_2.num_batches_tracked']
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the TATA model.
        
        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for the input
            
        Returns:
            Logits for stance classification
        """
        # Get features from the encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        features = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Use the same features for TAG, TAW, and DeBERTa (simplified approach)
        tag_features = features
        taw_features = features
        deberta_features = features
        
        # Apply attention encoding
        attention_encoded = self.attention_encoding(deberta_features)
        
        # Process TAG and TAW representations
        tag_processed = self.linear_tag(tag_features)
        taw_processed = self.linear_taw(taw_features)
        
        # Combine TAG and TAW
        combined = torch.cat([tag_processed, taw_processed], dim=1)  # [batch_size, 1536]
        
        # Apply linear layers
        hidden1 = self.linear_11(combined)  # [batch_size, 768]
        hidden2 = self.linear_2(hidden1)    # [batch_size, 768]
        hidden2_norm = self.batchnorm_2(hidden2)
        
        # Combine with attention
        attention_combined = torch.cat([hidden2_norm, attention_encoded], dim=1)  # [batch_size, 1536]
        hidden3 = self.linear_21(attention_combined)  # [batch_size, 768]
        
        # Final processing
        final_hidden = self.total_linear(attention_combined)  # [batch_size, 768]
        final_hidden = self.dropout(final_hidden)
        
        # Final classification
        logits = self.linear_3(final_hidden)  # [batch_size, 3]
        
        return logits


class StanceTATAScorer(StanceScorer):
    """
    TATA Stance Detection Scorer
    
    Implements the TATA (Topic-Agnostic and Topic-Aware) stance detection model
    for scoring target-passage pairs in the EFI corpus system.
    """
    
    def __init__(self, name: str = "tata", config: Optional[StanceTATAScorerConfig] = None):
        super().__init__(name, config)
        self.config = config or StanceTATAScorerConfig()
        
        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        if self.config.verbose:
            print(f"🔧 Initializing TATA scorer on device: {self.device}")
        
        # Load tokenizer with fallback options
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Failed to load {self.config.tokenizer_name}, trying fallback...")
            # Fallback to regular DeBERTa tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        
        # Load model (includes the encoder)
        self.model = self._load_model()
        
        if self.config.verbose:
            print(f"✅ TATA scorer initialized successfully")
    
    def _load_model(self) -> TATAModel:
        """Load the TATA model from the checkpoint file."""
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"TATA model not found at: {model_path}")
        
        if self.config.verbose:
            print(f"📥 Loading TATA model from: {model_path}")
        
        try:
            # Create a mock Object class for loading
            class Object:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            # Add Object class to the main module (where torch.load expects it)
            import __main__
            __main__.Object = Object
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                # Default config based on analysis
                from types import SimpleNamespace
                model_config = SimpleNamespace(
                    hidden_dropout_prob=0.3,
                    num_labels=3,
                    hidden_size=768
                )
            
            # Create model instance with state dict
            if 'model' in checkpoint:
                model = TATAModel(checkpoint['model'], model_config)
            else:
                raise ValueError("No model state dict found in checkpoint")
            
            # Set to evaluation mode
            model.eval()
            model.to(self.device)
            
            if self.config.verbose:
                print("✅ TATA model loaded successfully")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TATA model: {e}")
    
    def _preprocess_inputs(self, targets: List[str], passages: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess target-passage pairs for TATA model input.
        
        Args:
            targets: List of target topics/claims
            passages: List of texts expressing stance toward targets
            
        Returns:
            Dictionary containing tokenized inputs
        """
        # Combine target and passage for each pair
        combined_texts = []
        for target, passage in zip(targets, passages):
            # Format: "[CLS] target [SEP] passage [SEP]"
            combined_text = f"{target} [SEP] {passage}"
            combined_texts.append(combined_text)
        
        # Tokenize
        tokenized = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        return tokenized
    
    def _map_outputs(self, logits: torch.Tensor) -> List[Dict[str, float]]:
        """
        Map TATA model outputs to EFI stance format.
        
        Args:
            logits: Raw model logits [batch_size, 3]
            
        Returns:
            List of stance probability dictionaries
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Map TATA classes to EFI format
        # TATA: 0=against, 1=pro, 2=neutral
        # EFI: anti, pro, neutral, uncertain
        results = []
        
        for prob_row in probs:
            stance_scores = {
                "anti": float(prob_row[0]),      # against
                "pro": float(prob_row[1]),       # pro  
                "neutral": float(prob_row[2]),   # neutral
                "uncertain": 0.0                 # TATA doesn't have uncertain class
            }
            results.append(stance_scores)
        
        return results
    
    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """
        Score target-passage pairs for stance relationship using TATA model.
        
        Args:
            targets: List of target topics/claims
            passages: List of texts expressing stance toward targets
            
        Returns:
            List of dictionaries with stance class probabilities
        """
        if not passages:
            return []
        
        if self.config.verbose:
            print(f"🔬 Processing {len(targets)} stance pairs with TATA...")
        
        # Preprocess inputs
        tokenized = self._preprocess_inputs(targets, passages)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask']
            )
        
        # Map outputs to EFI format
        results = self._map_outputs(logits)
        
        if self.config.verbose:
            print(f"✅ Completed TATA stance scoring for {len(targets)} pairs")
        
        return results
