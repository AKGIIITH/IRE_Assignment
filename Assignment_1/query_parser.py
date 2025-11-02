import re
from typing import List, Union

class QueryNode:
    """Represents a node in the query parse tree."""
    pass

class TermNode(QueryNode):
    def __init__(self, term: str):
        self.term = term
    
    def __repr__(self):
        return f'Term("{self.term}")'

class PhraseNode(QueryNode):
    def __init__(self, terms: List[str]):
        self.terms = terms
    
    def __repr__(self):
        return f'Phrase({self.terms})'

class NotNode(QueryNode):
    def __init__(self, child: QueryNode):
        self.child = child
    
    def __repr__(self):
        return f'NOT({self.child})'

class AndNode(QueryNode):
    def __init__(self, left: QueryNode, right: QueryNode):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f'({self.left} AND {self.right})'

class OrNode(QueryNode):
    def __init__(self, left: QueryNode, right: QueryNode):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f'({self.left} OR {self.right})'

class QueryParser:
    """Parse boolean queries using Shunting-yard algorithm."""
    
    def __init__(self):
        self.precedence = {
            'OR': 1,
            'AND': 2,
            'NOT': 3,
            'PHRASE': 4
        }
    
    def tokenize(self, query: str) -> List[str]:
        """Tokenize the query string."""
        # Pattern to match: quoted strings, operators, parentheses
        pattern = r'"[^"]+"|AND|OR|NOT|\(|\)'
        tokens = re.findall(pattern, query)
        return tokens
    
    def parse(self, query: str) -> QueryNode:
        """Parse query string into AST."""
        tokens = self.tokenize(query)
        
        if not tokens:
            return TermNode("")
        
        output = []
        operators = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token.startswith('"') and token.endswith('"'):
                # Term or phrase
                content = token[1:-1]
                words = content.split()
                
                if len(words) > 1:
                    # It's a phrase
                    output.append(PhraseNode(words))
                else:
                    # Single term
                    output.append(TermNode(words[0] if words else ""))
            
            elif token == '(':
                operators.append(token)
            
            elif token == ')':
                # Pop until matching '('
                while operators and operators[-1] != '(':
                    self._apply_operator(output, operators.pop())
                
                if operators:
                    operators.pop()  # Remove '('
            
            elif token in ['AND', 'OR', 'NOT']:
                # Pop operators with higher or equal precedence
                while (operators and 
                       operators[-1] != '(' and 
                       operators[-1] in self.precedence and
                       self.precedence[operators[-1]] >= self.precedence[token]):
                    self._apply_operator(output, operators.pop())
                
                operators.append(token)
            
            i += 1
        
        # Pop remaining operators
        while operators:
            self._apply_operator(output, operators.pop())
        
        return output[0] if output else TermNode("")
    
    def _apply_operator(self, output: List[QueryNode], operator: str):
        """Apply an operator to the output stack."""
        if operator == 'NOT':
            if output:
                operand = output.pop()
                output.append(NotNode(operand))
        
        elif operator == 'AND':
            if len(output) >= 2:
                right = output.pop()
                left = output.pop()
                output.append(AndNode(left, right))
        
        elif operator == 'OR':
            if len(output) >= 2:
                right = output.pop()
                left = output.pop()
                output.append(OrNode(left, right))

# Simple usage example
if __name__ == "__main__":
    parser = QueryParser()
    
    test_queries = [
        '"apple"',
        '"apple" AND "banana"',
        '("apple" AND "banana") OR "orange"',
        '"apple" AND NOT "banana"',
        '"machine learning"',  # Phrase query
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        ast = parser.parse(query)
        print(f"AST: {ast}")