"""
Entity extraction for Sanskrit texts.
Identifies deities, sages, places, and concepts in Sanskrit quotes.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class SanskritEntityExtractor:
    """Extract entities from Sanskrit texts including deities, sages, places, and concepts."""
    
    def __init__(self):
        # Initialize entity dictionaries
        self.deities = self._load_deity_names()
        self.sages = self._load_sage_names()
        self.places = self._load_place_names()
        self.concepts = self._load_concept_terms()
        self.works = self._load_work_names()
        
        # Create pattern matchers
        self.deity_pattern = self._create_pattern(self.deities)
        self.sage_pattern = self._create_pattern(self.sages)
        self.place_pattern = self._create_pattern(self.places)
        self.concept_pattern = self._create_pattern(self.concepts)
        self.work_pattern = self._create_pattern(self.works)
    
    def _load_deity_names(self) -> Dict[str, List[str]]:
        """Load deity names and their variants."""
        return {
            'viṣṇu': ['viṣṇu', 'viṣṇo', 'viṣṇave', 'viṣṇor', 'hari', 'hare', 'nārāyaṇa', 
                      'keśava', 'mādhava', 'govinda', 'dāmodara', 'madhusūdana', 'vāsudeva',
                      'kṛṣṇa', 'kṛṣṇasya', 'murāri', 'janārdana'],
            'śiva': ['śiva', 'śivasya', 'śivāya', 'maheśvara', 'maheśa', 'śaṅkara', 
                     'rudra', 'īśa', 'īśvara', 'hara', 'bhava', 'śambhu', 'mahādeva',
                     'tryambaka', 'nīlakaṇṭha'],
            'devī': ['devī', 'devyā', 'ambā', 'umā', 'pārvatī', 'śakti', 'bhagavatī',
                     'durgā', 'kālī', 'lakṣmī', 'sarasvatī', 'gaurī', 'caṇḍī'],
            'brahmā': ['brahmā', 'brahmaṇaḥ', 'pitāmaha', 'svayambhū', 'prajāpati', 
                       'virañci', 'caturmukha', 'kamalāsana'],
            'gaṇeśa': ['gaṇeśa', 'gaṇapati', 'vināyaka', 'vighnēśvara', 'lambodara',
                       'gajavadana', 'heraṃba'],
            'indra': ['indra', 'indrasya', 'śakra', 'vāsava', 'purandara', 'devarāja',
                      'maghavan', 'sahasrākṣa'],
            'agni': ['agni', 'agneḥ', 'agnaye', 'vahni', 'anala', 'hutāśana', 'pāvaka'],
            'sūrya': ['sūrya', 'sūryasya', 'āditya', 'bhāskara', 'ravi', 'savitṛ',
                      'arka', 'mihira', 'bhānu'],
            'hanumān': ['hanumān', 'hanūmān', 'hanumataḥ', 'añjaneya', 'mārutātmaja',
                        'vāyuputra', 'pavana-tanaya'],
            'rāma': ['rāma', 'rāmasya', 'rāghava', 'raghunātha', 'dāśarathi', 'kausalyeya',
                     'sītāpati', 'raghūttama']
        }
    
    def _load_sage_names(self) -> Dict[str, List[str]]:
        """Load sage names and their variants."""
        return {
            'vyāsa': ['vyāsa', 'vyāsaḥ', 'vedavyāsa', 'bādarāyaṇa', 'kṛṣṇadvaipāyana'],
            'vālmīki': ['vālmīki', 'vālmīkeḥ', 'ādikavi', 'pracetasaḥ'],
            'vasiṣṭha': ['vasiṣṭha', 'vasiṣṭhasya', 'brahmarṣi'],
            'viśvāmitra': ['viśvāmitra', 'viśvāmitrasya', 'kauśika', 'gādheya'],
            'nārada': ['nārada', 'nāradasya', 'devarṣi', 'nāradamuni'],
            'agastya': ['agastya', 'agastyasya', 'agastimuni', 'kumbhayoni'],
            'bharadvāja': ['bharadvāja', 'bharadvājasya'],
            'atri': ['atri', 'atreḥ', 'atraye'],
            'aṅgiras': ['aṅgiras', 'aṅgirasaḥ', 'aṅgirā'],
            'gautama': ['gautama', 'gautamasya', 'gotama'],
            'jamadagni': ['jamadagni', 'jamadagneḥ'],
            'kaśyapa': ['kaśyapa', 'kaśyapasya'],
            'bhṛgu': ['bhṛgu', 'bhṛgoḥ', 'bhārgava'],
            'patañjali': ['patañjali', 'patañjaleḥ', 'yogasūtrakāra'],
            'kālidāsa': ['kālidāsa', 'kālidāsasya', 'kavikulaguru'],
            'śaṅkara': ['śaṅkara', 'śaṅkarācārya', 'ādiśaṅkara', 'bhagavatpāda']
        }
    
    def _load_place_names(self) -> Dict[str, List[str]]:
        """Load place names and their variants."""
        return {
            'ayodhyā': ['ayodhyā', 'ayodhyāyāṃ', 'sāketa'],
            'kāśī': ['kāśī', 'kāśyāṃ', 'vārāṇasī', 'benares'],
            'mathurā': ['mathurā', 'mathurāyāṃ', 'brajabhūmi'],
            'vṛndāvana': ['vṛndāvana', 'vṛndāvane', 'vraja', 'gokula'],
            'dvārakā': ['dvārakā', 'dvārakāyāṃ', 'dvāravatī'],
            'hastināpura': ['hastināpura', 'hastināpure', 'gajasāhvaya'],
            'laṅkā': ['laṅkā', 'laṅkāyāṃ', 'siṃhala'],
            'kurukṣetra': ['kurukṣetra', 'kurukṣetre', 'dharmakṣetra'],
            'prayāga': ['prayāga', 'prayāge', 'triveṇī', 'allahabad'],
            'gayā': ['gayā', 'gayāyāṃ', 'bodhagayā'],
            'himālaya': ['himālaya', 'himavat', 'himācala', 'himādri'],
            'gaṅgā': ['gaṅgā', 'gaṅgāyāṃ', 'bhāgīrathī', 'jāhnavī'],
            'yamunā': ['yamunā', 'yamunāyāṃ', 'kālindī'],
            'narmadā': ['narmadā', 'narmadāyāṃ', 'revā'],
            'godāvarī': ['godāvarī', 'godāvaryāṃ', 'gautamī']
        }
    
    def _load_concept_terms(self) -> Dict[str, List[str]]:
        """Load philosophical and religious concept terms."""
        return {
            'dharma': ['dharma', 'dharmasya', 'dharmāya', 'dharmāṇi'],
            'karma': ['karma', 'karmaṇā', 'karmaṇi', 'karmaphala'],
            'bhakti': ['bhakti', 'bhaktyā', 'bhaktimān'],
            'jñāna': ['jñāna', 'jñānena', 'jñānī'],
            'yoga': ['yoga', 'yogena', 'yogī', 'yogin'],
            'mokṣa': ['mokṣa', 'mokṣāya', 'mukti', 'nirvāṇa', 'kaivalya'],
            'saṃsāra': ['saṃsāra', 'saṃsāre', 'bhavacakra'],
            'brahman': ['brahman', 'brahmaṇi', 'parabrahma'],
            'ātman': ['ātman', 'ātmani', 'ātmā'],
            'māyā': ['māyā', 'māyayā', 'māyāvī'],
            'śānti': ['śānti', 'śāntyai', 'śāntimān'],
            'ahiṃsā': ['ahiṃsā', 'ahiṃsayā'],
            'satya': ['satya', 'satyena', 'satyavādin'],
            'tapas': ['tapas', 'tapasā', 'tapasvī'],
            'vairāgya': ['vairāgya', 'vairāgyeṇa', 'virāga']
        }
    
    def _load_work_names(self) -> Dict[str, List[str]]:
        """Load Sanskrit work names."""
        return {
            'mahābhārata': ['mahābhārata', 'mahābhārate', 'bhārata'],
            'rāmāyaṇa': ['rāmāyaṇa', 'rāmāyaṇe', 'ādikāvya'],
            'bhagavadgītā': ['bhagavadgītā', 'gītā', 'gītāyāṃ'],
            'bhāgavata': ['bhāgavata', 'bhāgavate', 'śrīmadbhāgavata'],
            'yogasūtra': ['yogasūtra', 'pātañjalayogasūtra'],
            'vedānta': ['vedānta', 'vedāntasūtra', 'brahmasūtra'],
            'manusmṛti': ['manusmṛti', 'manusmṛtau', 'mānavadharmaśāstra'],
            'arthaśāstra': ['arthaśāstra', 'kauṭilīya'],
            'kāmasūtra': ['kāmasūtra', 'vātsyāyana'],
            'upaniṣad': ['upaniṣad', 'upaniṣat', 'vedāntavākya']
        }
    
    def _create_pattern(self, entity_dict: Dict[str, List[str]]) -> re.Pattern:
        """Create regex pattern from entity dictionary."""
        all_variants = []
        for variants in entity_dict.values():
            all_variants.extend(variants)
        
        # Sort by length (descending) to match longer variants first
        all_variants.sort(key=len, reverse=True)
        
        # Escape special characters and create pattern
        escaped_variants = [re.escape(v) for v in all_variants]
        pattern = r'\b(' + '|'.join(escaped_variants) + r')\b'
        
        return re.compile(pattern, re.IGNORECASE)
    
    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, str, int, int]]]:
        """
        Extract entities from Sanskrit text.
        
        Args:
            text: Sanskrit text to extract entities from
            
        Returns:
            Dictionary mapping entity types to list of (entity, canonical_form, start, end)
        """
        entities = {
            'deities': [],
            'sages': [],
            'places': [],
            'concepts': [],
            'works': []
        }
        
        # Extract deities
        for match in self.deity_pattern.finditer(text):
            matched_text = match.group()
            canonical = self._get_canonical_form(matched_text, self.deities)
            entities['deities'].append((matched_text, canonical, match.start(), match.end()))
        
        # Extract sages
        for match in self.sage_pattern.finditer(text):
            matched_text = match.group()
            canonical = self._get_canonical_form(matched_text, self.sages)
            entities['sages'].append((matched_text, canonical, match.start(), match.end()))
        
        # Extract places
        for match in self.place_pattern.finditer(text):
            matched_text = match.group()
            canonical = self._get_canonical_form(matched_text, self.places)
            entities['places'].append((matched_text, canonical, match.start(), match.end()))
        
        # Extract concepts
        for match in self.concept_pattern.finditer(text):
            matched_text = match.group()
            canonical = self._get_canonical_form(matched_text, self.concepts)
            entities['concepts'].append((matched_text, canonical, match.start(), match.end()))
        
        # Extract works
        for match in self.work_pattern.finditer(text):
            matched_text = match.group()
            canonical = self._get_canonical_form(matched_text, self.works)
            entities['works'].append((matched_text, canonical, match.start(), match.end()))
        
        return entities
    
    def _get_canonical_form(self, matched_text: str, entity_dict: Dict[str, List[str]]) -> str:
        """Get canonical form of matched entity."""
        matched_lower = matched_text.lower()
        
        for canonical, variants in entity_dict.items():
            if matched_lower in [v.lower() for v in variants]:
                return canonical
        
        return matched_text
    
    def get_entity_summary(self, text: str) -> Dict[str, Set[str]]:
        """
        Get a summary of unique entities in the text.
        
        Returns:
            Dictionary mapping entity types to sets of canonical entity names
        """
        entities = self.extract_entities(text)
        
        summary = {
            'deities': set(),
            'sages': set(),
            'places': set(),
            'concepts': set(),
            'works': set()
        }
        
        for entity_type, entity_list in entities.items():
            for _, canonical, _, _ in entity_list:
                summary[entity_type].add(canonical)
        
        return summary
    
    def extract_entity_features(self, text: str) -> Dict[str, float]:
        """
        Extract entity-based features for ranking.
        
        Returns:
            Dictionary of feature names to values
        """
        entities = self.extract_entities(text)
        
        features = {
            'num_deities': len(entities['deities']),
            'num_sages': len(entities['sages']),
            'num_places': len(entities['places']),
            'num_concepts': len(entities['concepts']),
            'num_works': len(entities['works']),
            'total_entities': sum(len(v) for v in entities.values()),
            'unique_deities': len(set(e[1] for e in entities['deities'])),
            'unique_sages': len(set(e[1] for e in entities['sages'])),
            'unique_places': len(set(e[1] for e in entities['places'])),
            'unique_concepts': len(set(e[1] for e in entities['concepts'])),
            'unique_works': len(set(e[1] for e in entities['works']))
        }
        
        # Add density features
        text_length = len(text.split())
        if text_length > 0:
            features['entity_density'] = features['total_entities'] / text_length
            features['deity_density'] = features['num_deities'] / text_length
            features['concept_density'] = features['num_concepts'] / text_length
        else:
            features['entity_density'] = 0.0
            features['deity_density'] = 0.0
            features['concept_density'] = 0.0
        
        return features