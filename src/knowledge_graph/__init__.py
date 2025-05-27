"""
Knowledge Graph module for Sanskrit Quote Retrieval.

This module provides:
- Entity extraction and linking for Sanskrit texts
- KG-based query expansion
- Semantic filtering using KG paths
"""

from .entity_extractor import SanskritEntityExtractor
from .knowledge_base import SanskritKnowledgeBase
from .kg_enhancer import KGEnhancer

__all__ = ['SanskritEntityExtractor', 'SanskritKnowledgeBase', 'KGEnhancer']