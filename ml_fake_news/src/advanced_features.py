"""
Advanced feature extraction for fake news detection.
Includes linguistic and stylistic features that correlate with misinformation.
"""

import numpy as np
import pandas as pd
import re
from collections import Counter


def extract_linguistic_features(text):
    """
    Extract linguistic features that correlate with fake news:
    - Capitalization patterns
    - Punctuation frequency
    - Word length statistics
    - Sentence structure
    - Emotional language indicators
    """
    features = {}
    
    # Basic stats
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    if features['word_count'] == 0:
        # Return zero features for empty text
        return {k: 0.0 for k in get_feature_names()}
    
    # Capitalization features (indicator of shouting/emphasis)
    caps = sum(1 for c in text if c.isupper())
    features['capital_ratio'] = caps / len(text) if len(text) > 0 else 0
    
    words = text.split()
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    features['all_caps_word_ratio'] = all_caps_words / features['word_count'] if features['word_count'] > 0 else 0
    
    # Punctuation features
    exclamations = text.count('!')
    questions = text.count('?')
    periods = text.count('.')
    ellipsis = len(re.findall(r'\.{2,}', text))
    
    features['exclamation_count'] = exclamations
    features['question_count'] = questions
    features['ellipsis_count'] = ellipsis
    features['punctuation_intensity'] = (exclamations + questions * 0.5 + ellipsis * 0.5) / features['word_count']
    
    # Word length features
    word_lengths = [len(w.strip('.,!?;:')) for w in words if w.strip('.,!?;:')]
    if word_lengths:
        features['avg_word_length'] = np.mean(word_lengths)
        features['max_word_length'] = np.max(word_lengths)
        features['word_length_std'] = np.std(word_lengths) if len(word_lengths) > 1 else 0
    else:
        features['avg_word_length'] = 0
        features['max_word_length'] = 0
        features['word_length_std'] = 0
    
    # Sentence features
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features['sentence_count'] = len(sentences)
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if sentences else 0
    
    # Emotional/sensational language indicators
    emotional_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 'exclusive', 
                       'exposed', 'shocking', 'scandal', 'outrage', 'devastating']
    emotional_count = sum(text.lower().count(word) for word in emotional_words)
    features['emotional_word_count'] = emotional_count
    features['emotional_word_ratio'] = emotional_count / features['word_count'] if features['word_count'] > 0 else 0
    
    # Question marks (often used in fake headlines)
    features['question_ratio'] = questions / features['word_count'] if features['word_count'] > 0 else 0
    
    # Numbers and metrics (can indicate false precision)
    numbers = len(re.findall(r'\b\d+\b', text))
    features['number_count'] = numbers
    features['number_ratio'] = numbers / features['word_count'] if features['word_count'] > 0 else 0
    
    # Duplicate words (sign of low quality)
    word_counts = Counter(words)
    max_word_freq = max(word_counts.values()) if word_counts else 0
    features['max_word_frequency'] = max_word_freq / features['word_count'] if features['word_count'] > 0 else 0
    
    return features


def get_feature_names():
    """Get all linguistic feature names"""
    return [
        'text_length', 'word_count', 'capital_ratio', 'all_caps_word_ratio',
        'exclamation_count', 'question_count', 'ellipsis_count', 'punctuation_intensity',
        'avg_word_length', 'max_word_length', 'word_length_std',
        'sentence_count', 'avg_sentence_length',
        'emotional_word_count', 'emotional_word_ratio',
        'question_ratio', 'number_count', 'number_ratio', 'max_word_frequency'
    ]


def extract_features_dataframe(texts):
    """Extract features for a list of texts and return as DataFrame"""
    features_list = []
    for text in texts:
        features = extract_linguistic_features(str(text))
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def combine_tfidf_and_linguistic(tfidf_matrix, linguistic_features):
    """
    Combine TF-IDF matrix with linguistic features.
    TF-IDF is sparse, so convert to dense and scale properly.
    """
    from sklearn.preprocessing import StandardScaler
    
    # Convert sparse matrix to dense if needed
    if hasattr(tfidf_matrix, 'toarray'):
        tfidf_dense = tfidf_matrix.toarray()
    else:
        tfidf_dense = tfidf_matrix
    
    # Normalize linguistic features
    scaler = StandardScaler()
    linguistic_scaled = scaler.fit_transform(linguistic_features)
    
    # Combine both feature sets
    combined = np.hstack([tfidf_dense, linguistic_scaled])
    
    return combined, scaler, linguistic_features.columns.tolist()
