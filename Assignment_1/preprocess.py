import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
from typing import Iterable
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> list[str]:
    """
    Preprocess text by tokenizing, lowercasing, removing stop words, and stemming.
    
    Args:
        text: Input text string
        
    Returns:
        List of preprocessed tokens
    """
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and keep only alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stem tokens
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def tokenize_without_preprocessing(text: str) -> list[str]:
    """Tokenize text without preprocessing for comparison."""
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

def generate_word_frequency_plots(documents: Iterable[str], output_prefix: str = 'word_freq'):
    """
    Generate word frequency plots before and after preprocessing.
    
    Args:
        documents: Iterable of document text strings
        output_prefix: Prefix for output plot filenames
    """
    print("Generating word frequency plots...")
    
    # Collect words before and after preprocessing
    words_before = []
    words_after = []
    
    for doc in documents:
        words_before.extend(tokenize_without_preprocessing(doc))
        words_after.extend(preprocess(doc))
    
    # Count frequencies
    freq_before = Counter(words_before)
    freq_after = Counter(words_after)
    
    # Get top 30 words
    top_before = freq_before.most_common(30)
    top_after = freq_after.most_common(30)
    
    # Plot before preprocessing
    plt.figure(figsize=(12, 6))
    words_b, counts_b = zip(*top_before)
    plt.bar(range(len(words_b)), counts_b)
    plt.xticks(range(len(words_b)), words_b, rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency (Before Preprocessing)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_before.png')
    plt.close()
    
    # Plot after preprocessing
    plt.figure(figsize=(12, 6))
    words_a, counts_a = zip(*top_after)
    plt.bar(range(len(words_a)), counts_a)
    plt.xticks(range(len(words_a)), words_a, rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency (After Preprocessing)')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_after.png')
    plt.close()
    
    print(f"Plots saved: {output_prefix}_before.png, {output_prefix}_after.png")
    print(f"Total unique words before: {len(freq_before)}")
    print(f"Total unique words after: {len(freq_after)}")