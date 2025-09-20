"""
Text preprocessing module for IRE assignment.
Handles tokenization, stopword removal, stemming, and word frequency analysis.
"""

import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def tokenize(self, text):
        """Tokenize text into words."""
        if not text:
            return []
        
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokens."""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text, remove_stopwords=True, apply_stemming=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            apply_stemming (bool): Whether to apply stemming
            
        Returns:
            list: Processed tokens
        """
        tokens = self.tokenize(text)
        
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def get_word_frequencies(self, texts, preprocessed=True):
        """
        Get word frequency distribution from a list of texts.
        
        Args:
            texts (list): List of text strings
            preprocessed (bool): Whether to apply preprocessing
            
        Returns:
            Counter: Word frequency counter
        """
        all_tokens = []
        
        for text in texts:
            if preprocessed:
                tokens = self.preprocess_text(text)
            else:
                tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        return Counter(all_tokens)
    
    def plot_word_frequencies(self, texts, top_n=20, save_path=None):
        """
        Plot word frequency comparison with and without preprocessing.
        
        Args:
            texts (list): List of text strings
            top_n (int): Number of top words to plot
            save_path (str): Optional path to save the plot
        """
        # Get frequencies with and without preprocessing
        freq_raw = self.get_word_frequencies(texts, preprocessed=False)
        freq_processed = self.get_word_frequencies(texts, preprocessed=True)
        
        # Get top words
        top_raw = freq_raw.most_common(top_n)
        top_processed = freq_processed.most_common(top_n)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot raw frequencies
        words_raw, counts_raw = zip(*top_raw)
        ax1.bar(range(len(words_raw)), counts_raw)
        ax1.set_title('Word Frequencies (Raw)')
        ax1.set_xlabel('Words')
        ax1.set_ylabel('Frequency')
        ax1.set_xticks(range(len(words_raw)))
        ax1.set_xticklabels(words_raw, rotation=45, ha='right')
        
        # Plot processed frequencies
        words_processed, counts_processed = zip(*top_processed)
        ax2.bar(range(len(words_processed)), counts_processed)
        ax2.set_title('Word Frequencies (Preprocessed)')
        ax2.set_xlabel('Words')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(len(words_processed)))
        ax2.set_xticklabels(words_processed, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return freq_raw, freq_processed

if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Information retrieval is a fascinating field of computer science.",
        "Elasticsearch is a powerful search engine built on Apache Lucene."
    ]
    
    print("Testing text preprocessing...")
    for text in sample_texts:
        processed = preprocessor.preprocess_text(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()
    
    # Generate frequency plots
    print("Generating word frequency plots...")
    preprocessor.plot_word_frequencies(sample_texts)
