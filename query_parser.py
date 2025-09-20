"""
Boolean query parser for IRE assignment.
Supports AND, OR, NOT, PHRASE operations with proper precedence.
"""

import re
from typing import List, Union, Dict, Any

class Token:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class QueryParser:
    def __init__(self):
        # Define token patterns
        self.patterns = [
            (r'\(', 'LPAREN'),
            (r'\)', 'RPAREN'),
            (r'\bAND\b', 'AND'),
            (r'\bOR\b', 'OR'),
            (r'\bNOT\b', 'NOT'),
            (r'"[^"]*"', 'PHRASE'),  # Quoted phrases
            (r'"[^"]*', 'TERM'),     # Single quoted terms
            (r'[A-Za-z0-9_]+', 'TERM'),  # Regular terms
            (r'\s+', 'WHITESPACE'),
        ]
        
        self.compiled_patterns = [(re.compile(pattern), token_type) 
                                 for pattern, token_type in self.patterns]
    
    def tokenize(self, query: str) -> List[Token]:
        """
        Tokenize the query string.
        
        Args:
            query (str): Query string to tokenize
            
        Returns:
            List[Token]: List of tokens
        """
        tokens = []
        pos = 0
        
        while pos < len(query):
            matched = False
            
            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(query, pos)
                if match:
                    value = match.group(0)
                    if token_type != 'WHITESPACE':  # Skip whitespace tokens
                        tokens.append(Token(token_type, value))
                    pos = match.end()
                    matched = True
                    break
            
            if not matched:
                # Skip unknown characters
                pos += 1
        
        return tokens
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse a boolean query string into an AST.
        
        Grammar:
        QUERY    := EXPR
        EXPR     := TERM | (EXPR) | EXPR AND EXPR | EXPR OR EXPR | NOT EXPR | PHRASE
        
        Precedence (highest to lowest): PHRASE, NOT, AND, OR
        
        Args:
            query (str): Query string to parse
            
        Returns:
            Dict: Abstract syntax tree representation
        """
        tokens = self.tokenize(query)
        if not tokens:
            return {'type': 'empty'}
        
        self.tokens = tokens
        self.pos = 0
        return self.parse_or_expr()
    
    def current_token(self) -> Union[Token, None]:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def consume_token(self, expected_type: str = None) -> Token:
        """Consume and return current token."""
        token = self.current_token()
        if token and (expected_type is None or token.type == expected_type):
            self.pos += 1
            return token
        return None
    
    def parse_or_expr(self) -> Dict[str, Any]:
        """Parse OR expressions (lowest precedence)."""
        left = self.parse_and_expr()
        
        while self.current_token() and self.current_token().type == 'OR':
            self.consume_token('OR')
            right = self.parse_and_expr()
            left = {
                'type': 'or',
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_and_expr(self) -> Dict[str, Any]:
        """Parse AND expressions."""
        left = self.parse_not_expr()
        
        while self.current_token() and self.current_token().type == 'AND':
            self.consume_token('AND')
            right = self.parse_not_expr()
            left = {
                'type': 'and',
                'left': left,
                'right': right
            }
        
        return left
    
    def parse_not_expr(self) -> Dict[str, Any]:
        """Parse NOT expressions."""
        if self.current_token() and self.current_token().type == 'NOT':
            self.consume_token('NOT')
            expr = self.parse_primary()
            return {
                'type': 'not',
                'expr': expr
            }
        
        return self.parse_primary()
    
    def parse_primary(self) -> Dict[str, Any]:
        """Parse primary expressions (terms, phrases, parentheses)."""
        token = self.current_token()
        
        if not token:
            return {'type': 'empty'}
        
        if token.type == 'LPAREN':
            self.consume_token('LPAREN')
            expr = self.parse_or_expr()
            self.consume_token('RPAREN')
            return expr
        
        elif token.type == 'PHRASE':
            phrase = self.consume_token('PHRASE').value
            # Remove quotes
            phrase_content = phrase.strip('"')
            return {
                'type': 'phrase',
                'value': phrase_content
            }
        
        elif token.type == 'TERM':
            term = self.consume_token('TERM').value
            # Remove quotes if present
            term_content = term.strip('"')
            return {
                'type': 'term',
                'value': term_content
            }
        
        else:
            # Skip unknown tokens
            self.consume_token()
            return self.parse_primary()

class QueryExecutor:
    def __init__(self, index):
        """
        Initialize query executor with an index.
        
        Args:
            index: Index object that supports search operations
        """
        self.index = index
    
    def execute(self, ast: Dict[str, Any]) -> set:
        """
        Execute a parsed query AST against the index.
        
        Args:
            ast (Dict): Abstract syntax tree from parser
            
        Returns:
            set: Set of document IDs matching the query
        """
        if ast['type'] == 'term':
            return self.index.search_term(ast['value'])
        
        elif ast['type'] == 'phrase':
            return self.index.search_phrase(ast['value'])
        
        elif ast['type'] == 'and':
            left_results = self.execute(ast['left'])
            right_results = self.execute(ast['right'])
            return left_results.intersection(right_results)
        
        elif ast['type'] == 'or':
            left_results = self.execute(ast['left'])
            right_results = self.execute(ast['right'])
            return left_results.union(right_results)
        
        elif ast['type'] == 'not':
            expr_results = self.execute(ast['expr'])
            all_docs = self.index.get_all_document_ids()
            return all_docs - expr_results
        
        elif ast['type'] == 'empty':
            return set()
        
        else:
            return set()

if __name__ == "__main__":
    # Example usage
    parser = QueryParser()
    
    test_queries = [
        '"information retrieval"',
        'information AND retrieval',
        'search OR query',
        'NOT elasticsearch',
        '(information AND retrieval) OR (search AND engine)',
        '"machine learning" AND (python OR java) AND NOT deprecated'
    ]
    
    print("Testing query parser...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            ast = parser.parse(query)
            print(f"AST: {ast}")
        except Exception as e:
            print(f"Error parsing query: {e}")
    
    # Test tokenizer
    print("\n\nTesting tokenizer...")
    test_query = '(information AND retrieval) OR "machine learning"'
    tokens = parser.tokenize(test_query)
    print(f"Query: {test_query}")
    print(f"Tokens: {tokens}")
