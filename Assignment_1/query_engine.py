from typing import List, Set, Dict, Tuple
from query_parser import *

class QueryEngine:
    """Execute parsed queries against the inverted index."""
    
    def __init__(self, index: Dict, doc_count: int, use_skip_pointers: bool = False):
        """
        Args:
            index: The inverted index dictionary
            doc_count: Total number of documents
            use_skip_pointers: Whether to use skip pointers for AND operations
        """
        self.index = index
        self.doc_count = doc_count
        self.use_skip_pointers = use_skip_pointers
    
    def execute(self, query_node: QueryNode, mode: str = 'TAAT') -> List[Tuple[str, float]]:
        """
        Execute query and return results.
        
        Args:
            query_node: Parsed query AST
            mode: 'TAAT' (Term-at-a-Time) or 'DAAT' (Document-at-a-Time)
            
        Returns:
            List of (doc_id, score) tuples
        """
        if mode == 'TAAT':
            return self._execute_taat(query_node)
        else:
            return self._execute_daat(query_node)
    
    def _execute_taat(self, node: QueryNode) -> List[Tuple[str, float]]:
        """Term-at-a-Time execution."""
        result_set = self._evaluate_node_taat(node)
        
        # Convert set to list with scores
        return [(doc_id, 1.0) for doc_id in result_set]
    
    def _evaluate_node_taat(self, node: QueryNode) -> Set[str]:
        """Recursively evaluate query node in TAAT mode."""
        if isinstance(node, TermNode):
            return self._get_term_docs(node.term)
        
        elif isinstance(node, PhraseNode):
            return self._phrase_query(node.terms)
        
        elif isinstance(node, NotNode):
            all_docs = set(self.index.get('__all_docs__', []))
            child_docs = self._evaluate_node_taat(node.child)
            return all_docs - child_docs
        
        elif isinstance(node, AndNode):
            left_docs = self._evaluate_node_taat(node.left)
            right_docs = self._evaluate_node_taat(node.right)
            return self._intersect(left_docs, right_docs)
        
        elif isinstance(node, OrNode):
            left_docs = self._evaluate_node_taat(node.left)
            right_docs = self._evaluate_node_taat(node.right)
            return left_docs | right_docs
        
        return set()
    
    def _execute_daat(self, node: QueryNode) -> List[Tuple[str, float]]:
        """Document-at-a-Time execution."""
        # Collect all terms and their posting lists
        term_postings = self._collect_term_postings(node)
        
        if not term_postings:
            return []
        
        # Score documents
        doc_scores = {}
        
        for term, postings in term_postings.items():
            for doc_id, score, positions in postings:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += score
        
        # Sort by score
        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return results
    
    def _collect_term_postings(self, node: QueryNode) -> Dict[str, List]:
        """Collect all term postings from the query tree."""
        postings = {}
        
        if isinstance(node, TermNode):
            term = node.term.lower()
            if term in self.index:
                postings[term] = self.index[term]
        
        elif isinstance(node, (AndNode, OrNode)):
            postings.update(self._collect_term_postings(node.left))
            postings.update(self._collect_term_postings(node.right))
        
        elif isinstance(node, NotNode):
            postings.update(self._collect_term_postings(node.child))
        
        return postings
    
    def _get_term_docs(self, term: str) -> Set[str]:
        """Get document IDs containing the term."""
        term = term.lower()
        if term not in self.index:
            return set()
        
        postings = self.index[term]
        return {doc_id for doc_id, *_ in postings}
    
    def _intersect(self, list1: Set[str], list2: Set[str]) -> Set[str]:
        """Intersect two document sets."""
        return list1 & list2
    
    def _phrase_query(self, terms: List[str]) -> Set[str]:
        """Find documents containing the phrase."""
        if not terms:
            return set()
        
        # Get postings for all terms
        terms = [t.lower() for t in terms]
        term_postings = []
        
        for term in terms:
            if term not in self.index:
                return set()
            term_postings.append(self.index[term])
        
        # Find documents containing all terms
        doc_sets = [set(doc_id for doc_id, *_ in postings) for postings in term_postings]
        candidate_docs = set.intersection(*doc_sets)
        
        # Check for phrase in candidate documents
        result_docs = set()
        
        for doc_id in candidate_docs:
            # Get positions for each term in this document
            term_positions = []
            for i, term in enumerate(terms):
                for doc, *rest in term_postings[i]:
                    if doc == doc_id:
                        positions = rest[-1] if rest else []
                        term_positions.append(set(positions))
                        break
            
            # Check if terms appear consecutively
            if self._check_phrase_positions(term_positions):
                result_docs.add(doc_id)
        
        return result_docs
    
    def _check_phrase_positions(self, term_positions: List[Set[int]]) -> bool:
        """Check if term positions form a phrase."""
        if not term_positions:
            return False
        
        # For each position of the first term
        for pos in term_positions[0]:
            match = True
            # Check if subsequent terms appear at consecutive positions
            for i in range(1, len(term_positions)):
                if (pos + i) not in term_positions[i]:
                    match = False
                    break
            
            if match:
                return True
        
        return False