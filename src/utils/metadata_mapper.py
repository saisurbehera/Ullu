"""
Metadata mapper for Sanskrit corpus.
Maps source filenames to proper work names and authors.
"""

import re
from typing import Dict, Optional

class SanskritMetadataMapper:
    """Maps source file names to work metadata."""
    
    def __init__(self):
        # Mapping of filename patterns to work information
        self.work_mappings = {
            # Puranas
            'brndp': {'work': 'Brahmanda Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'mkp': {'work': 'Markandeya Purana', 'author': 'Markandeya', 'category': 'Purana'},
            'skp': {'work': 'Skanda Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'bhp': {'work': 'Bhagavata Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'vp': {'work': 'Vishnu Purana', 'author': 'Parashara', 'category': 'Purana'},
            'ap': {'work': 'Agni Purana', 'author': 'Agni', 'category': 'Purana'},
            'gp': {'work': 'Garuda Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'pp': {'work': 'Padma Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'sp': {'work': 'Shiva Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'mp': {'work': 'Matsya Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'lp': {'work': 'Linga Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'kp': {'work': 'Kurma Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'vamp': {'work': 'Vamana Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'varp': {'work': 'Varaha Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'brvp': {'work': 'Brahmavaivarta Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'bvp': {'work': 'Bhavishya Purana', 'author': 'Vyasa', 'category': 'Purana'},
            
            # Epics
            'mbh': {'work': 'Mahabharata', 'author': 'Vyasa', 'category': 'Itihasa'},
            'ram': {'work': 'Ramayana', 'author': 'Valmiki', 'category': 'Itihasa'},
            
            # Vedas
            'rv': {'work': 'Rig Veda', 'author': 'Various Rishis', 'category': 'Veda'},
            'av': {'work': 'Atharva Veda', 'author': 'Atharvan', 'category': 'Veda'},
            'yv': {'work': 'Yajur Veda', 'author': 'Various Rishis', 'category': 'Veda'},
            'sv': {'work': 'Sama Veda', 'author': 'Various Rishis', 'category': 'Veda'},
            'athv': {'work': 'Atharvaveda Samhita', 'author': 'Atharvan', 'category': 'Veda'},
            
            # Upanishads
            'isha': {'work': 'Isha Upanishad', 'author': 'Unknown', 'category': 'Upanishad'},
            'kena': {'work': 'Kena Upanishad', 'author': 'Unknown', 'category': 'Upanishad'},
            'katha': {'work': 'Katha Upanishad', 'author': 'Unknown', 'category': 'Upanishad'},
            'pras': {'work': 'Prashna Upanishad', 'author': 'Unknown', 'category': 'Upanishad'},
            'mund': {'work': 'Mundaka Upanishad', 'author': 'Unknown', 'category': 'Upanishad'},
            'mand': {'work': 'Mandukya Upanishad', 'author': 'Unknown', 'category': 'Upanishad'},
            
            # Shastras
            'ms': {'work': 'Manusmriti', 'author': 'Manu', 'category': 'Dharmashastra'},
            'ys': {'work': 'Yajnavalkya Smriti', 'author': 'Yajnavalkya', 'category': 'Dharmashastra'},
            
            # Other texts
            'bhg': {'work': 'Bhagavad Gita', 'author': 'Krishna/Vyasa', 'category': 'Philosophy'},
            'devi': {'work': 'Devi Mahatmya', 'author': 'Markandeya', 'category': 'Stotra'},
            'ss': {'work': 'Surya Siddhanta', 'author': 'Unknown', 'category': 'Astronomy'},
            'ch': {'work': 'Charaka Samhita', 'author': 'Charaka', 'category': 'Ayurveda'},
            'su': {'work': 'Sushruta Samhita', 'author': 'Sushruta', 'category': 'Ayurveda'},
        }
        
        # Additional patterns for complex filenames
        self.pattern_mappings = {
            r'.*athv.*': {'work': 'Atharvaveda Samhita', 'author': 'Atharvan', 'category': 'Veda'},
            r'.*rkv.*': {'work': 'Rig Veda Khila', 'author': 'Various Rishis', 'category': 'Veda'},
            r'.*narp.*': {'work': 'Narada Purana', 'author': 'Narada', 'category': 'Purana'},
            r'.*nsp.*': {'work': 'Narasimha Purana', 'author': 'Unknown', 'category': 'Purana'},
        }
    
    def extract_metadata(self, source_file: str, category: str = None) -> Dict[str, str]:
        """Extract work, author, and category from source filename."""
        if not source_file:
            return {'work': 'Unknown', 'author': 'Unknown', 'category': category or 'Unknown'}
        
        # Clean filename - remove extensions and paths
        clean_filename = source_file.lower().replace('.htm', '').replace('.html', '')
        clean_filename = re.sub(r'[0-9\-_]', '', clean_filename)
        
        # Try direct mappings first
        for pattern, metadata in self.work_mappings.items():
            if pattern in clean_filename:
                result = metadata.copy()
                if category and category != '1_sanskr':
                    result['category'] = self._normalize_category(category)
                return result
        
        # Try regex patterns
        for pattern, metadata in self.pattern_mappings.items():
            if re.match(pattern, clean_filename):
                result = metadata.copy()
                if category and category != '1_sanskr':
                    result['category'] = self._normalize_category(category)
                return result
        
        # Extract work from citation patterns in text (like BndP_3,1.1)
        work_from_citation = self._extract_from_citation(source_file)
        if work_from_citation:
            return work_from_citation
        
        # Default fallback
        return {
            'work': self._guess_work_from_filename(clean_filename),
            'author': 'Unknown',
            'category': self._normalize_category(category) if category else 'Sanskrit Text'
        }
    
    def _extract_from_citation(self, text: str) -> Optional[Dict[str, str]]:
        """Extract work info from citation patterns like BndP_3,1.1."""
        citation_patterns = {
            'BndP': {'work': 'Brahmanda Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'MarkP': {'work': 'Markandeya Purana', 'author': 'Markandeya', 'category': 'Purana'},
            'SkP': {'work': 'Skanda Purana', 'author': 'Vyasa', 'category': 'Purana'},
            'NarP': {'work': 'Narada Purana', 'author': 'Narada', 'category': 'Purana'},
            'NsP': {'work': 'Narasimha Purana', 'author': 'Unknown', 'category': 'Purana'},
            'RKV': {'work': 'Rig Veda Khila', 'author': 'Various Rishis', 'category': 'Veda'},
        }
        
        for pattern, metadata in citation_patterns.items():
            if pattern in text:
                return metadata
                
        return None
    
    def _normalize_category(self, category: str) -> str:
        """Normalize category names."""
        if not category or category == '1_sanskr':
            return 'Sanskrit Text'
        
        category_map = {
            '1_sanskr': 'Sanskrit Text',
            'purana': 'Purana',
            'veda': 'Veda', 
            'upanishad': 'Upanishad',
            'itihasa': 'Itihasa',
            'shastra': 'Shastra'
        }
        
        return category_map.get(category.lower(), category.title())
    
    def _guess_work_from_filename(self, filename: str) -> str:
        """Make an educated guess about work from filename."""
        if 'purana' in filename or 'p_' in filename:
            return 'Purana Text'
        elif 'veda' in filename or 'v_' in filename:
            return 'Vedic Text'
        elif 'upanishad' in filename:
            return 'Upanishad'
        elif 'gita' in filename:
            return 'Bhagavad Gita'
        elif 'ramayana' in filename or 'ram' in filename:
            return 'Ramayana'
        elif 'mahabharata' in filename or 'mbh' in filename:
            return 'Mahabharata'
        else:
            return 'Sanskrit Text'


# Global instance for use in app
metadata_mapper = SanskritMetadataMapper()