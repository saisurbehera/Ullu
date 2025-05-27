"""
Knowledge Graph enhancer for improving Sanskrit quote retrieval.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import logging

from .entity_extractor import SanskritEntityExtractor
from .knowledge_base import SanskritKnowledgeBase

logger = logging.getLogger(__name__)

class KGEnhancer:
    """
    Enhance retrieval using knowledge graph relationships and entity analysis.
    """
    
    def __init__(self, kb: Optional[SanskritKnowledgeBase] = None, 
                 extractor: Optional[SanskritEntityExtractor] = None):
        self.kb = kb or SanskritKnowledgeBase()
        self.extractor = extractor or SanskritEntityExtractor()
    
    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand query using KG relationships.
        
        Args:
            query: Original query text
            max_expansions: Maximum number of query expansions
            
        Returns:
            List of expanded queries including original
        """
        expanded_queries = [query]
        
        # Extract entities from query
        entities = self.extractor.get_entity_summary(query)
        
        # Expand based on entity relationships
        for entity_type, entity_set in entities.items():
            for entity in entity_set:
                # Get related entities
                related = self.kb.get_related_entities(entity)
                
                for relation, related_entity in related[:max_expansions]:
                    # Create expanded query by adding related entity
                    if relation in ['avatar_of', 'has_avatar', 'also_known_as']:
                        # Replace entity with related entity
                        expanded_query = query.replace(entity, related_entity)
                        if expanded_query != query:
                            expanded_queries.append(expanded_query)
                    elif relation in ['consort_of', 'devotee_of', 'student_of']:
                        # Add related entity to query
                        expanded_query = f"{query} {related_entity}"
                        expanded_queries.append(expanded_query)
        
        # Expand based on work associations
        if entities.get('deities') or entities.get('sages'):
            all_entities = entities.get('deities', set()) | entities.get('sages', set())
            common_works = self.kb.find_common_works(list(all_entities))
            
            for work in list(common_works)[:max_expansions]:
                work_info = self.kb.get_work_info(work)
                if work_info:
                    expanded_query = f"{query} {work_info['title']}"
                    expanded_queries.append(expanded_query)
        
        return list(set(expanded_queries))  # Remove duplicates
    
    def compute_kg_features(self, query: str, passage: str) -> Dict[str, float]:
        """
        Compute KG-based features for query-passage pair.
        
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Extract entities from query and passage
        query_entities = self.extractor.get_entity_summary(query)
        passage_entities = self.extractor.get_entity_summary(passage)
        
        # Entity overlap features
        for entity_type in ['deities', 'sages', 'places', 'concepts', 'works']:
            query_set = query_entities.get(entity_type, set())
            passage_set = passage_entities.get(entity_type, set())
            
            # Exact match overlap
            overlap = len(query_set & passage_set)
            features[f'{entity_type}_overlap'] = overlap
            
            # Jaccard similarity
            if query_set or passage_set:
                jaccard = len(query_set & passage_set) / len(query_set | passage_set)
                features[f'{entity_type}_jaccard'] = jaccard
            else:
                features[f'{entity_type}_jaccard'] = 0.0
        
        # Related entity features
        related_overlap = self._compute_related_entity_overlap(query_entities, passage_entities)
        features.update(related_overlap)
        
        # Work association features
        work_features = self._compute_work_association_features(query_entities, passage_entities)
        features.update(work_features)
        
        # Path-based features
        path_features = self._compute_path_features(query_entities, passage_entities)
        features.update(path_features)
        
        return features
    
    def _compute_related_entity_overlap(self, query_entities: Dict, passage_entities: Dict) -> Dict[str, float]:
        """Compute overlap considering related entities."""
        features = {}
        
        # Get all entities
        query_all = set()
        passage_all = set()
        
        for entities in query_entities.values():
            query_all.update(entities)
        
        for entities in passage_entities.values():
            passage_all.update(entities)
        
        # Check if passage entities are related to query entities
        related_count = 0
        total_relations = 0
        
        for q_entity in query_all:
            related_entities = set()
            for _, r_entity in self.kb.get_related_entities(q_entity):
                related_entities.add(r_entity)
            
            if related_entities:
                total_relations += len(related_entities)
                related_count += len(related_entities & passage_all)
        
        features['related_entity_overlap'] = related_count
        features['related_entity_ratio'] = related_count / max(total_relations, 1)
        
        return features
    
    def _compute_work_association_features(self, query_entities: Dict, passage_entities: Dict) -> Dict[str, float]:
        """Compute features based on work associations."""
        features = {}
        
        # Get all entities
        query_all = set()
        passage_all = set()
        
        for entities in query_entities.values():
            query_all.update(entities)
        
        for entities in passage_entities.values():
            passage_all.update(entities)
        
        # Find common works
        query_works = set()
        for entity in query_all:
            query_works.update(self.kb.get_works_by_entity(entity))
        
        passage_works = set()
        for entity in passage_all:
            passage_works.update(self.kb.get_works_by_entity(entity))
        
        # Also add explicitly mentioned works
        query_works.update(query_entities.get('works', set()))
        passage_works.update(passage_entities.get('works', set()))
        
        # Compute features
        features['common_works'] = len(query_works & passage_works)
        features['work_overlap_ratio'] = len(query_works & passage_works) / max(len(query_works | passage_works), 1)
        
        # Check if entities typically co-occur in same works
        if query_all and passage_all:
            common_works = self.kb.find_common_works(list(query_all | passage_all))
            features['entity_cooccurrence_works'] = len(common_works)
        else:
            features['entity_cooccurrence_works'] = 0
        
        return features
    
    def _compute_path_features(self, query_entities: Dict, passage_entities: Dict) -> Dict[str, float]:
        """Compute path-based features between entities."""
        features = {}
        
        # Get all entities
        query_all = list(set().union(*query_entities.values()))
        passage_all = list(set().union(*passage_entities.values()))
        
        if not query_all or not passage_all:
            features['min_path_length'] = 10  # Large value for no path
            features['avg_path_length'] = 10
            features['connected_pairs'] = 0
            return features
        
        # Compute shortest paths
        path_lengths = []
        connected_pairs = 0
        
        for q_entity in query_all[:5]:  # Limit to prevent too many computations
            for p_entity in passage_all[:5]:
                path = self.kb.get_path_between_entities(q_entity, p_entity, max_depth=3)
                
                if path is not None:
                    path_lengths.append(len(path))
                    connected_pairs += 1
        
        if path_lengths:
            features['min_path_length'] = min(path_lengths)
            features['avg_path_length'] = np.mean(path_lengths)
        else:
            features['min_path_length'] = 10
            features['avg_path_length'] = 10
        
        features['connected_pairs'] = connected_pairs
        
        return features
    
    def filter_by_kg_consistency(self, query: str, candidates: List[Dict], 
                                 threshold: float = 0.3) -> List[Dict]:
        """
        Filter candidates based on KG consistency.
        
        Args:
            query: Query text
            candidates: List of candidate passages with metadata
            threshold: Minimum KG consistency score
            
        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []
        
        for candidate in candidates:
            # Compute KG consistency score
            consistency_score = self._compute_kg_consistency(query, candidate)
            
            if consistency_score >= threshold:
                candidate['kg_consistency_score'] = consistency_score
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _compute_kg_consistency(self, query: str, candidate: Dict) -> float:
        """Compute KG consistency score for a candidate."""
        passage = candidate.get('text', '')
        metadata = candidate.get('metadata', {})
        
        # Extract entities
        query_entities = self.extractor.get_entity_summary(query)
        passage_entities = self.extractor.get_entity_summary(passage)
        
        scores = []
        
        # Check work consistency
        if metadata.get('work'):
            work_id = metadata['work'].lower().replace(' ', '')
            expected_entities = self.kb.get_entities_in_work(work_id)
            
            # Check if query entities are expected in this work
            query_all = set().union(*query_entities.values())
            if query_all and expected_entities:
                work_consistency = len(query_all & expected_entities) / len(query_all)
                scores.append(work_consistency)
        
        # Check entity relationship consistency
        kg_features = self.compute_kg_features(query, passage)
        
        # Normalize features to [0, 1]
        if kg_features.get('related_entity_ratio', 0) > 0:
            scores.append(kg_features['related_entity_ratio'])
        
        if kg_features.get('work_overlap_ratio', 0) > 0:
            scores.append(kg_features['work_overlap_ratio'])
        
        if kg_features.get('min_path_length', 10) < 10:
            # Convert path length to similarity (shorter = better)
            path_score = 1.0 / (1 + kg_features['min_path_length'])
            scores.append(path_score)
        
        # Return average of all scores
        if scores:
            return np.mean(scores)
        else:
            return 0.0
    
    def enhance_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Enhance candidates with KG features and rerank.
        
        Args:
            query: Query text
            candidates: List of candidate passages
            
        Returns:
            Enhanced and reranked candidates
        """
        enhanced_candidates = []
        
        for candidate in candidates:
            # Add KG features
            passage = candidate.get('text', '')
            kg_features = self.compute_kg_features(query, passage)
            
            # Add features to candidate
            if 'features' not in candidate:
                candidate['features'] = {}
            candidate['features'].update(kg_features)
            
            # Compute KG score
            kg_score = self._compute_kg_score(kg_features)
            candidate['kg_score'] = kg_score
            
            # Combine with original score
            original_score = candidate.get('score', 0.0)
            candidate['combined_score'] = 0.7 * original_score + 0.3 * kg_score
            
            enhanced_candidates.append(candidate)
        
        # Sort by combined score
        enhanced_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return enhanced_candidates
    
    def _compute_kg_score(self, features: Dict[str, float]) -> float:
        """Compute overall KG score from features."""
        # Weight different feature types
        weights = {
            'deities_overlap': 2.0,
            'sages_overlap': 1.5,
            'works_overlap': 2.5,
            'concepts_overlap': 1.0,
            'places_overlap': 1.0,
            'related_entity_ratio': 1.5,
            'work_overlap_ratio': 2.0,
            'entity_cooccurrence_works': 1.5,
            'connected_pairs': 1.0
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature, value in features.items():
            if feature in weights:
                weighted_sum += weights[feature] * value
                total_weight += weights[feature]
        
        if total_weight > 0:
            # Normalize to [0, 1]
            score = weighted_sum / total_weight
            return min(1.0, score)
        else:
            return 0.0
    
    def get_entity_context(self, entity: str) -> Dict[str, Any]:
        """Get contextual information about an entity from KG."""
        context = {
            'entity': entity,
            'relations': self.kb.get_related_entities(entity),
            'works': list(self.kb.get_works_by_entity(entity)),
            'related_concepts': []
        }
        
        # Find related concepts
        if entity in self.kb.concept_hierarchy:
            context['related_concepts'] = self.kb.concept_hierarchy[entity]
        
        # Get work details
        context['work_details'] = []
        for work_id in context['works']:
            work_info = self.kb.get_work_info(work_id)
            if work_info:
                context['work_details'].append(work_info)
        
        return context