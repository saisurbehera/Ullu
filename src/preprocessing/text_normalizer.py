import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import logging

logger = logging.getLogger(__name__)

class SanskritTextNormalizer:
    """
    Normalizes Sanskrit text across different scripts and encoding schemes.
    Handles Devanagari, IAST, HK (Harvard-Kyoto), and other common schemes.
    """
    
    def __init__(self):
        self.vedic_accents = {
            '\u0951': '',  # udatta
            '\u0952': '',  # anudatta
            '\u1cda': '',  # vedic tone karshana
            '\u1cdb': '',  # vedic tone shara
            '\u1cdc': '',  # vedic tone prenkha
        }
        
        self.diacritic_map = {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ṝ': 'r',
            'ḷ': 'l', 'ḹ': 'l', 'ē': 'e', 'ō': 'o', 'ṃ': 'm',
            'ḥ': 'h', 'ṅ': 'n', 'ñ': 'n', 'ṭ': 't', 'ḍ': 'd',
            'ṇ': 'n', 'ś': 's', 'ṣ': 's'
        }
    
    def normalize_to_iast(self, text: str) -> str:
        """Convert text from any supported script to IAST."""
        # Detect script
        script = self._detect_script(text)
        
        if script == 'devanagari':
            normalized = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        elif script == 'hk':
            normalized = transliterate(text, sanscript.HK, sanscript.IAST)
        elif script == 'itrans':
            normalized = transliterate(text, sanscript.ITRANS, sanscript.IAST)
        elif script == 'slp1':
            normalized = transliterate(text, sanscript.SLP1, sanscript.IAST)
        else:
            normalized = text  # Already in IAST or unknown
        
        # Apply NFC normalization
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Remove vedic accents
        for accent, replacement in self.vedic_accents.items():
            normalized = normalized.replace(accent, replacement)
        
        return normalized.strip()
    
    def create_ascii_index_form(self, text: str) -> str:
        """Create ASCII version for fuzzy matching."""
        iast_text = self.normalize_to_iast(text)
        
        # Replace diacritics with base characters
        ascii_form = iast_text
        for diacritic, base in self.diacritic_map.items():
            ascii_form = ascii_form.replace(diacritic, base)
        
        # Remove remaining non-ASCII characters
        ascii_form = ''.join(c for c in ascii_form if ord(c) < 128)
        
        return ascii_form.lower()
    
    def _detect_script(self, text: str) -> str:
        """Detect the script/encoding of the input text."""
        # Count character types
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
        iast_count = len(re.findall(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]', text))
        hk_count = len(re.findall(r'[AEIOURLM~H]', text))
        
        total_chars = len(text)
        
        if devanagari_count > total_chars * 0.1:
            return 'devanagari'
        elif iast_count > 2:
            return 'iast'
        elif hk_count > total_chars * 0.1 and text.isupper():
            return 'hk'
        elif re.search(r'[~\^]', text):
            return 'itrans'
        elif re.search(r'[xyz]', text.lower()):
            return 'slp1'
        else:
            return 'unknown'
    
    def clean_text(self, text: str) -> str:
        """Clean and standardize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers and punctuation that might interfere
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[।॥]+', ' ', text)  # Remove dandas
        
        # Remove HTML/XML artifacts
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        return text.strip()
    
    def process_text(self, text: str) -> Dict[str, str]:
        """Process text and return multiple normalized forms."""
        cleaned = self.clean_text(text)
        iast = self.normalize_to_iast(cleaned)
        ascii_form = self.create_ascii_index_form(iast)
        
        return {
            'original': text,
            'cleaned': cleaned,
            'iast': iast,
            'ascii': ascii_form,
            'length': len(iast.split())
        }

class SandhiSplitter:
    """
    Rule-based Sanskrit sandhi splitter.
    Implements common sandhi rules for splitting compound words.
    """
    
    def __init__(self):
        self.vowel_sandhi_rules = [
            # a + a = ā
            (r'ā(?=[aāiīuūṛṝḷḹeē])', 'a a'),
            # a + i = e
            (r'e(?=[bcdfghjklmnpqrstvwxyz])', 'a i'),
            # a + u = o
            (r'o(?=[bcdfghjklmnpqrstvwxyz])', 'a u'),
        ]
        
        self.consonant_clusters = [
            'kṣ', 'jñ', 'tr', 'dv', 'tv', 'sv', 'śv', 'sth', 'spr', 'str'
        ]
    
    def split_compound(self, word: str) -> List[str]:
        """Attempt to split a compound word."""
        if len(word) < 6:  # Too short to be compound
            return [word]
        
        # Simple heuristic splitting
        splits = self._find_split_points(word)
        
        if splits:
            return splits
        else:
            return [word]
    
    def _find_split_points(self, word: str) -> List[str]:
        """Find potential split points in a word."""
        # Look for common prefixes/suffixes
        prefixes = ['pra', 'upa', 'anu', 'apa', 'ava', 'adhi', 'abhi', 'sam', 'pari']
        suffixes = ['bhāva', 'tva', 'tā', 'maya', 'vat', 'mat']
        
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                remainder = word[len(prefix):]
                return [prefix, remainder]
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                base = word[:-len(suffix)]
                return [base, suffix]
        
        return []
    
    def process_text(self, text: str) -> List[str]:
        """Split text into words and process compounds."""
        words = text.split()
        processed_words = []
        
        for word in words:
            splits = self.split_compound(word)
            processed_words.extend(splits)
        
        return processed_words

class MorphologicalAnalyzer:
    """
    Basic morphological analysis for Sanskrit.
    This is a simplified version - in production, would use UoH tagger or similar.
    """
    
    def __init__(self):
        self.verb_endings = {
            'ति': {'person': '3', 'number': 'sg', 'voice': 'active'},
            'ते': {'person': '3', 'number': 'sg', 'voice': 'middle'},
            'न्ति': {'person': '3', 'number': 'pl', 'voice': 'active'},
            'न्ते': {'person': '3', 'number': 'pl', 'voice': 'middle'},
        }
        
        self.noun_endings = {
            'अः': {'case': 'nom', 'number': 'sg'},
            'आः': {'case': 'nom', 'number': 'pl'},
            'अम्': {'case': 'acc', 'number': 'sg'},
            'आन्': {'case': 'acc', 'number': 'pl'},
            'एन': {'case': 'inst', 'number': 'sg'},
        }
    
    def analyze_word(self, word: str) -> Dict:
        """Analyze a single word morphologically."""
        # Convert to Devanagari for analysis if needed
        if not re.search(r'[\u0900-\u097F]', word):
            word_dev = transliterate(word, sanscript.IAST, sanscript.DEVANAGARI)
        else:
            word_dev = word
        
        analysis = {
            'word': word,
            'lemma': self._extract_lemma(word),
            'pos': self._guess_pos(word),
            'features': {}
        }
        
        # Check for verb endings
        for ending, features in self.verb_endings.items():
            if word_dev.endswith(ending):
                analysis['pos'] = 'verb'
                analysis['features'].update(features)
                break
        
        # Check for noun endings
        for ending, features in self.noun_endings.items():
            if word_dev.endswith(ending):
                analysis['pos'] = 'noun'
                analysis['features'].update(features)
                break
        
        return analysis
    
    def _extract_lemma(self, word: str) -> str:
        """Extract the lemma (crude approximation)."""
        # Remove common endings
        common_endings = ['ति', 'ते', 'न्ति', 'न्ते', 'अः', 'आः', 'अम्', 'आन्']
        
        if not re.search(r'[\u0900-\u097F]', word):
            word_dev = transliterate(word, sanscript.IAST, sanscript.DEVANAGARI)
        else:
            word_dev = word
        
        for ending in common_endings:
            if word_dev.endswith(ending):
                return word_dev[:-len(ending)]
        
        return word_dev
    
    def _guess_pos(self, word: str) -> str:
        """Guess part of speech based on patterns."""
        if word.endswith(('ति', 'ते')):
            return 'verb'
        elif word.endswith(('अः', 'आः', 'अम्')):
            return 'noun'
        elif word in ['च', 'वा', 'अथवा', 'किन्तु']:
            return 'conj'
        else:
            return 'unknown'
    
    def process_text(self, words: List[str]) -> List[Dict]:
        """Analyze a list of words."""
        return [self.analyze_word(word) for word in words]

class SanskritPreprocessor:
    """Main preprocessing pipeline combining all components."""
    
    def __init__(self):
        self.normalizer = SanskritTextNormalizer()
        self.sandhi_splitter = SandhiSplitter()
        self.morphological_analyzer = MorphologicalAnalyzer()
    
    def preprocess(self, text: str) -> Dict:
        """Complete preprocessing pipeline."""
        # Step 1: Normalize text
        normalized = self.normalizer.process_text(text)
        
        # Step 2: Split compounds
        split_words = self.sandhi_splitter.process_text(normalized['iast'])
        
        # Step 3: Morphological analysis
        morphological_analysis = self.morphological_analyzer.process_text(split_words)
        
        # Extract lemmas for indexing
        lemmas = [analysis['lemma'] for analysis in morphological_analysis]
        lemmas_text = ' '.join(lemmas)
        
        return {
            'original': text,
            'normalized': normalized,
            'split_words': split_words,
            'morphological_analysis': morphological_analysis,
            'lemmas': lemmas,
            'lemmas_text': lemmas_text,
            'ascii_form': normalized['ascii']
        }

def main():
    """Test the preprocessing pipeline."""
    preprocessor = SanskritPreprocessor()
    
    test_texts = [
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः",
        "sarvadharman parityajya mam ekam saranam vraja",
        "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत"
    ]
    
    for text in test_texts:
        result = preprocessor.preprocess(text)
        print(f"Original: {result['original']}")
        print(f"IAST: {result['normalized']['iast']}")
        print(f"ASCII: {result['ascii_form']}")
        print(f"Lemmas: {result['lemmas_text']}")
        print("-" * 50)

if __name__ == "__main__":
    main()