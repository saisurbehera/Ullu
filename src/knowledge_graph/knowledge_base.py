"""
Sanskrit Knowledge Base for managing relationships between entities, works, and concepts.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SanskritKnowledgeBase:
    """
    Knowledge base containing relationships between Sanskrit entities, works, and concepts.
    """
    
    def __init__(self):
        self.entities = {}  # entity_id -> entity info
        self.works = {}  # work_id -> work info
        self.relations = defaultdict(list)  # (entity1, relation, entity2)
        self.entity_to_works = defaultdict(set)  # entity -> set of works
        self.work_to_entities = defaultdict(set)  # work -> set of entities
        self.concept_hierarchy = {}  # concept -> parent concepts
        
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with core Sanskrit knowledge."""
        # Add major works
        self._add_works()
        
        # Add entity relationships
        self._add_entity_relations()
        
        # Add concept hierarchy
        self._add_concept_hierarchy()
        
        # Add entity-work associations
        self._add_entity_work_associations()
    
    def _add_works(self):
        """Add major Sanskrit works to the knowledge base."""
        self.works = {
            'mahābhārata': {
                'id': 'mahābhārata',
                'title': 'Mahābhārata',
                'author': 'vyāsa',
                'type': 'epic',
                'books': ['ādiparvan', 'sabhāparvan', 'vanaparvan', 'virāṭaparvan', 
                          'udyogaparvan', 'bhīṣmaparvan', 'droṇaparvan', 'karṇaparvan',
                          'śalyaparvan', 'sauptikaparvan', 'strīparvan', 'śāntiparvan',
                          'anuśāsanaparvan', 'aśvamedhikaparvan', 'āśramavāsikaparvan',
                          'mausalaparvan', 'mahāprasthānikaparvan', 'svargārohaṇaparvan'],
                'era': 'classical',
                'language': 'sanskrit'
            },
            'rāmāyaṇa': {
                'id': 'rāmāyaṇa',
                'title': 'Rāmāyaṇa',
                'author': 'vālmīki',
                'type': 'epic',
                'books': ['bālakāṇḍa', 'ayodhyākāṇḍa', 'araṇyakāṇḍa', 
                          'kiṣkindhākāṇḍa', 'sundarakāṇḍa', 'yuddhakāṇḍa', 'uttarakāṇḍa'],
                'era': 'classical',
                'language': 'sanskrit'
            },
            'bhagavadgītā': {
                'id': 'bhagavadgītā',
                'title': 'Bhagavadgītā',
                'author': 'vyāsa',
                'type': 'philosophical',
                'parent_work': 'mahābhārata',
                'book': 'bhīṣmaparvan',
                'chapters': 18,
                'era': 'classical',
                'language': 'sanskrit'
            },
            'bhāgavata': {
                'id': 'bhāgavata',
                'title': 'Śrīmad Bhāgavata Purāṇa',
                'author': 'vyāsa',
                'type': 'purāṇa',
                'books': 12,
                'era': 'classical',
                'language': 'sanskrit',
                'focus': 'viṣṇu'
            },
            'yogasūtra': {
                'id': 'yogasūtra',
                'title': 'Yoga Sūtra',
                'author': 'patañjali',
                'type': 'philosophical',
                'chapters': ['samādhipāda', 'sādhanapāda', 'vibhūtipāda', 'kaivalyapāda'],
                'era': 'classical',
                'language': 'sanskrit'
            },
            'manusmṛti': {
                'id': 'manusmṛti',
                'title': 'Manusmṛti',
                'author': 'manu',
                'type': 'dharmaśāstra',
                'chapters': 12,
                'era': 'classical',
                'language': 'sanskrit'
            },
            'meghadūta': {
                'id': 'meghadūta',
                'title': 'Meghadūta',
                'author': 'kālidāsa',
                'type': 'kāvya',
                'era': 'classical',
                'language': 'sanskrit'
            },
            'kumārasambhava': {
                'id': 'kumārasambhava',
                'title': 'Kumārasambhava',
                'author': 'kālidāsa',
                'type': 'kāvya',
                'era': 'classical',
                'language': 'sanskrit'
            },
            'raghuvaṃśa': {
                'id': 'raghuvaṃśa',
                'title': 'Raghuvaṃśa',
                'author': 'kālidāsa',
                'type': 'kāvya',
                'era': 'classical',
                'language': 'sanskrit'
            },
            'abhijñānaśākuntala': {
                'id': 'abhijñānaśākuntala',
                'title': 'Abhijñānaśākuntala',
                'author': 'kālidāsa',
                'type': 'nāṭaka',
                'era': 'classical',
                'language': 'sanskrit'
            }
        }
    
    def _add_entity_relations(self):
        """Add relationships between entities."""
        # Family relations
        self.add_relation('kṛṣṇa', 'avatar_of', 'viṣṇu')
        self.add_relation('rāma', 'avatar_of', 'viṣṇu')
        self.add_relation('hanumān', 'devotee_of', 'rāma')
        self.add_relation('arjuna', 'disciple_of', 'kṛṣṇa')
        self.add_relation('pārvatī', 'consort_of', 'śiva')
        self.add_relation('sītā', 'consort_of', 'rāma')
        self.add_relation('rādhā', 'beloved_of', 'kṛṣṇa')
        self.add_relation('lakṣmī', 'consort_of', 'viṣṇu')
        self.add_relation('sarasvatī', 'consort_of', 'brahmā')
        
        # Guru-disciple relations
        self.add_relation('arjuna', 'student_of', 'droṇa')
        self.add_relation('rāma', 'student_of', 'vasiṣṭha')
        self.add_relation('kṛṣṇa', 'student_of', 'sāndīpani')
        
        # Location associations
        self.add_relation('kṛṣṇa', 'born_in', 'mathurā')
        self.add_relation('kṛṣṇa', 'lived_in', 'vṛndāvana')
        self.add_relation('kṛṣṇa', 'ruled', 'dvārakā')
        self.add_relation('rāma', 'born_in', 'ayodhyā')
        self.add_relation('rāma', 'ruled', 'ayodhyā')
        
        # Author relations
        self.add_relation('vyāsa', 'author_of', 'mahābhārata')
        self.add_relation('vyāsa', 'author_of', 'bhāgavata')
        self.add_relation('vālmīki', 'author_of', 'rāmāyaṇa')
        self.add_relation('kālidāsa', 'author_of', 'meghadūta')
        self.add_relation('kālidāsa', 'author_of', 'kumārasambhava')
        self.add_relation('patañjali', 'author_of', 'yogasūtra')
    
    def _add_concept_hierarchy(self):
        """Add hierarchical relationships between concepts."""
        self.concept_hierarchy = {
            'puruṣārtha': ['dharma', 'artha', 'kāma', 'mokṣa'],
            'yoga': ['jñānayoga', 'bhaktiyoga', 'karmayoga', 'rājayoga'],
            'guṇa': ['sattva', 'rajas', 'tamas'],
            'varṇa': ['brāhmaṇa', 'kṣatriya', 'vaiśya', 'śūdra'],
            'āśrama': ['brahmacārya', 'gṛhastha', 'vānaprastha', 'sannyāsa'],
            'pramāṇa': ['pratyakṣa', 'anumāna', 'śabda', 'upamāna', 'arthāpatti', 'anupalabdhi']
        }
    
    def _add_entity_work_associations(self):
        """Add associations between entities and works they appear in."""
        # Mahābhārata entities
        mahābhārata_entities = {
            'kṛṣṇa', 'arjuna', 'bhīma', 'yudhiṣṭhira', 'nakula', 'sahadeva',
            'draupadī', 'karṇa', 'duryodhana', 'bhīṣma', 'droṇa', 'vyāsa'
        }
        for entity in mahābhārata_entities:
            self.add_entity_work_association(entity, 'mahābhārata')
        
        # Rāmāyaṇa entities
        rāmāyaṇa_entities = {
            'rāma', 'sītā', 'lakṣmaṇa', 'hanumān', 'rāvaṇa', 'bharata',
            'daśaratha', 'kauśalyā', 'sugrīva', 'vālī', 'vibhīṣaṇa'
        }
        for entity in rāmāyaṇa_entities:
            self.add_entity_work_association(entity, 'rāmāyaṇa')
        
        # Bhagavadgītā entities
        bhagavadgītā_entities = {'kṛṣṇa', 'arjuna'}
        for entity in bhagavadgītā_entities:
            self.add_entity_work_association(entity, 'bhagavadgītā')
        
        # Bhāgavata entities
        bhāgavata_entities = {
            'kṛṣṇa', 'rādhā', 'yaśodā', 'nanda', 'kaṃsa', 'balarāma',
            'rukmiṇī', 'satyabhāmā', 'uddhava'
        }
        for entity in bhāgavata_entities:
            self.add_entity_work_association(entity, 'bhāgavata')
    
    def add_relation(self, entity1: str, relation: str, entity2: str):
        """Add a relation between two entities."""
        self.relations[(entity1, relation)].append(entity2)
        
        # Add inverse relations for bidirectional queries
        inverse_relations = {
            'avatar_of': 'has_avatar',
            'devotee_of': 'has_devotee',
            'disciple_of': 'has_disciple',
            'consort_of': 'consort_of',
            'student_of': 'teacher_of',
            'born_in': 'birthplace_of',
            'lived_in': 'residence_of',
            'ruled': 'ruled_by',
            'author_of': 'authored_by',
            'beloved_of': 'beloved_of'
        }
        
        if relation in inverse_relations:
            inverse_rel = inverse_relations[relation]
            self.relations[(entity2, inverse_rel)].append(entity1)
    
    def add_entity_work_association(self, entity: str, work: str):
        """Add association between entity and work."""
        self.entity_to_works[entity].add(work)
        self.work_to_entities[work].add(entity)
    
    def get_related_entities(self, entity: str, relation: Optional[str] = None) -> List[Tuple[str, str]]:
        """Get entities related to the given entity."""
        if relation:
            return [(relation, e) for e in self.relations.get((entity, relation), [])]
        else:
            # Get all relations for this entity
            results = []
            for (e, r), targets in self.relations.items():
                if e == entity:
                    for target in targets:
                        results.append((r, target))
            return results
    
    def get_works_by_entity(self, entity: str) -> Set[str]:
        """Get works where the entity appears."""
        return self.entity_to_works.get(entity, set())
    
    def get_entities_in_work(self, work: str) -> Set[str]:
        """Get entities that appear in the work."""
        return self.work_to_entities.get(work, set())
    
    def get_work_info(self, work_id: str) -> Optional[Dict]:
        """Get information about a work."""
        return self.works.get(work_id)
    
    def find_common_works(self, entities: List[str]) -> Set[str]:
        """Find works that contain all given entities."""
        if not entities:
            return set()
        
        common_works = self.get_works_by_entity(entities[0])
        for entity in entities[1:]:
            common_works &= self.get_works_by_entity(entity)
        
        return common_works
    
    def get_path_between_entities(self, entity1: str, entity2: str, max_depth: int = 3) -> Optional[List[Tuple[str, str, str]]]:
        """
        Find path between two entities in the knowledge graph.
        Returns list of (entity, relation, entity) tuples.
        """
        if entity1 == entity2:
            return []
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(entity1, [])])
        visited = {entity1}
        
        while queue and len(visited) < 1000:  # Limit to prevent infinite loops
            current, path = queue.popleft()
            
            if len(path) >= max_depth:
                continue
            
            # Get all related entities
            for relation, target in self.get_related_entities(current):
                if target not in visited:
                    new_path = path + [(current, relation, target)]
                    
                    if target == entity2:
                        return new_path
                    
                    visited.add(target)
                    queue.append((target, new_path))
        
        return None
    
    def get_concept_descendants(self, concept: str) -> Set[str]:
        """Get all descendant concepts in the hierarchy."""
        descendants = set()
        
        if concept in self.concept_hierarchy:
            for child in self.concept_hierarchy[concept]:
                descendants.add(child)
                descendants.update(self.get_concept_descendants(child))
        
        return descendants
    
    def save(self, path: str):
        """Save knowledge base to file."""
        kb_data = {
            'entities': self.entities,
            'works': self.works,
            'relations': {f"{k[0]}|{k[1]}": v for k, v in self.relations.items()},
            'entity_to_works': {k: list(v) for k, v in self.entity_to_works.items()},
            'work_to_entities': {k: list(v) for k, v in self.work_to_entities.items()},
            'concept_hierarchy': self.concept_hierarchy
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(kb_data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load knowledge base from file."""
        with open(path, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        self.entities = kb_data['entities']
        self.works = kb_data['works']
        
        # Reconstruct relations
        self.relations.clear()
        for key, values in kb_data['relations'].items():
            entity, relation = key.split('|')
            self.relations[(entity, relation)] = values
        
        # Reconstruct sets
        self.entity_to_works = defaultdict(set)
        for k, v in kb_data['entity_to_works'].items():
            self.entity_to_works[k] = set(v)
        
        self.work_to_entities = defaultdict(set)
        for k, v in kb_data['work_to_entities'].items():
            self.work_to_entities[k] = set(v)
        
        self.concept_hierarchy = kb_data['concept_hierarchy']