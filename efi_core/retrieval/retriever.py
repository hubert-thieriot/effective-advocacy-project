"""
Unified retriever for vector search over embedded corpora and libraries.

This module provides a Retriever ABC that can work with both EmbeddedCorpus
and EmbeddedLibrary to perform fast similarity search using either FAISS
indexes or brute-force cosine similarity.
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import logging
from abc import ABC, abstractmethod

from efi_core.types import ChunkerSpec, EmbedderSpec, Candidate

logger = logging.getLogger(__name__)






    
