"""
Simple text normalizer fallback when indic-transliteration is not available.
Provides basic Sanskrit text processing without external dependencies.
"""

import re
import unicodedata
from typing import List, Dict, Optional

class SimpleTextNormalizer:
    """Simple fallback normalizer for Sanskrit text."""
    
    def __init__(self):
        # Basic diacritic normalization including Vedic accents
        self.diacritic_map = {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ṝ': 'r',
            'ḷ': 'l', 'ḹ': 'l', 'ē': 'e', 'ō': 'o', 'ṃ': 'm',
            'ḥ': 'h', 'ṅ': 'n', 'ñ': 'n', 'ṭ': 't', 'ḍ': 'd',
            'ṇ': 'n', 'ś': 's', 'ṣ': 's'
        }
        
        # Vedic accent marks to remove
        self.vedic_accents = {
            '\u0951': '',  # udatta (sáṃ)
            '\u0952': '',  # anudatta 
            '\u1cda': '',  # vedic tone karshana
            '\u1cdb': '',  # vedic tone shara
            '\u1cdc': '',  # vedic tone prenkha
            'á': 'a',      # acute accent
            'à': 'a',      # grave accent
            'é': 'e',      # acute accent
            'è': 'e',      # grave accent
            'í': 'i',      # acute accent
            'ì': 'i',      # grave accent
            'ó': 'o',      # acute accent
            'ò': 'o',      # grave accent
            'ú': 'u',      # acute accent
            'ù': 'u',      # grave accent
        }
        
        # Common Sanskrit word variations
        self.word_variants = {
            'dharma': ['dharma', 'dharm', 'righteous', 'duty'],
            'karma': ['karma', 'karm', 'action', 'deed'],
            'moksha': ['moksha', 'moksa', 'liberation', 'release'],
            'artha': ['artha', 'wealth', 'purpose', 'meaning'],
            'kama': ['kama', 'desire', 'pleasure', 'love']
        }
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization with Vedic accent handling."""
        if not text:
            return ""
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFC', text.lower())
        
        # Remove Vedic accent marks first
        for accented, replacement in self.vedic_accents.items():
            normalized = normalized.replace(accented, replacement)
        
        # Remove diacritics
        for accented, simple in self.diacritic_map.items():
            normalized = normalized.replace(accented, simple)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text."""
        result = text
        for accented, simple in self.diacritic_map.items():
            result = result.replace(accented, simple)
        return result
    
    def expand_query_terms(self, query: str) -> List[str]:
        """Expand query with Sanskrit term variations."""
        terms = [query.lower()]
        
        # Add version without diacritics
        no_diacritics = self.remove_diacritics(query.lower())
        if no_diacritics != query.lower():
            terms.append(no_diacritics)
        
        # Add known variants
        for sanskrit_term, variants in self.word_variants.items():
            if sanskrit_term in query.lower():
                terms.extend(variants)
        
        return list(set(terms))