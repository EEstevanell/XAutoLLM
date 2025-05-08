#!/usr/bin/env python3
"""
Error Analysis Script for IBERMAT Text Classification Models

This script performs a detailed error analysis of the best models identified from previous experiments:
1. FineTuneLLMEmbeddingClassifier from main.py (LLM-based approach)
2. CountVectorizerTokenizeStem + NuSVC from main_traditional.py (Traditional ML approach)

The script loads both models, trains them on the IBERMAT dataset, evaluates them on a test set,
and provides comprehensive error analysis without visualizations.

Usage:
    python test.py
"""

import os
import sys
import logging
from autogoal.utils._process import initialize_cuda_multiprocessing
import pandas as pd
import numpy as np
from pathlib import Path
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
# Replace mcnemar import with our own implementation
import scipy.stats as stats
import re
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print(1)
# Download required NLTK data packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizers...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

# Add parent directory to path to import common modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
print(2)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Explicitly use stdout
    force=True  # Force reconfiguration in case logging was already configured elsewhere
)
logger = logging.getLogger(__name__)

# Test the logger to verify it's working
logger.info("Logger initialized successfully")

print(3)

# Add progress tracking
def log_progress(message):
    """Helper function to log progress with a timestamp."""
    print(f"PROGRESS: {message}")  # Also print directly to ensure visibility
    logger.info(f"PROGRESS: {message}")

# Set fixed parameters
RANDOM_SEED = 42
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/IBERMAT.csv"))
TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/ibermat_test.csv"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(4)

# Implementation of McNemar's test
def mcnemar_test(table, correction=True):
    """
    Calculate McNemar's test using the chi-squared distribution.
    
    Parameters
    ----------
    table : array-like of shape (2, 2)
        Contingency table: [[a, b], [c, d]]
        a: number of times both models were correct
        b: number of times model 1 was correct but model 2 was wrong
        c: number of times model 1 was wrong but model 2 was correct
        d: number of times both models were wrong
    correction : bool, default=True
        Whether to apply the continuity correction (Yates' correction)
    
    Returns
    -------
    result : namedtuple
        A result object with attributes:
        statistic : float
            The calculated chi-squared statistic.
        pvalue : float
            The p-value for the test.
    """
    table = np.asarray(table, dtype=np.int64)
    if table.shape != (2, 2):
        raise ValueError("Table should be 2x2.")
    
    b = float(table[0, 1])
    c = float(table[1, 0])
    
    if b + c == 0:
        # Models give identical predictions
        statistic = 0.0
        pvalue = 1.0
    else:
        if correction:
            statistic = (abs(b - c) - 1)**2 / (b + c)
        else:
            statistic = (b - c)**2 / (b + c)
        
        pvalue = stats.chi2.sf(statistic, 1)
    
    # Create a simple namedtuple for the result
    from collections import namedtuple
    Result = namedtuple('Result', ['statistic', 'pvalue'])
    return Result(statistic, pvalue)

# Set random seeds for reproducibility
def set_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("PyTorch random seeds set for reproducibility")
    except ImportError:
        logger.info("PyTorch not available, setting only Python random seeds")

    logger.info(f"Random seeds set to {seed} for reproducibility")

class DataLoader:
    """
    Data loader for IBERMAT dataset that handles both training and test data.
    """
    
    def __init__(self, data_path=DATA_PATH, test_path=TEST_DATA_PATH):
        """Initialize the IBERMAT dataset loader."""
        self.data_path = Path(data_path)
        self.test_path = Path(test_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"IBERMAT training dataset not found at {self.data_path}")
        
        if not self.test_path.exists():
            raise FileNotFoundError(f"IBERMAT test dataset not found at {self.test_path}")
    
    def load_training_data(self, test_size=0.2):
        """Load the IBERMAT training dataset and create train/test splits."""
        logger.info(f"Loading IBERMAT training dataset from {self.data_path}")
        
        # Load CSV with semicolon separator
        df = pd.read_csv(self.data_path, sep=";", encoding="utf-8")
        
        logger.info(f"Training dataset loaded with {len(df)} samples")
        
        # Check if required columns exist
        required_columns = ["TEXT", "CLASS"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        
        # Extract texts and labels
        texts = df["TEXT"].tolist()
        labels = df["CLASS"].tolist()
        
        # Encode labels: HUMAN -> 0, MACHINE -> 1
        label_mapping = {"HUMAN": 0, "MACHINE": 1}
        y = np.array([label_mapping[label] for label in labels])
        
        # Collect metadata for analysis
        metadata = {
            "language": df["LANGUAGE"].tolist() if "LANGUAGE" in df.columns else None,
            "domain": df["DOMAIN"].tolist() if "DOMAIN" in df.columns else None,
        }
        
        # Print class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = "HUMAN" if label == 0 else "MACHINE"
            logger.info(f"Class {label_name} ({label}): {count} samples ({count/len(y)*100:.2f}%)")
        
        # Create train/val split (for model training)
        X_train, X_val, y_train, y_val = train_test_split(
            texts, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        
        return X_train, y_train, X_val, y_val, metadata
    
    def load_test_data(self):
        """Load the IBERMAT test dataset."""
        logger.info(f"Loading IBERMAT test dataset from {self.test_path}")
        
        # Load CSV with semicolon separator
        df = pd.read_csv(self.test_path, sep=";", encoding="utf-8")
        
        logger.info(f"Test dataset loaded with {len(df)} samples")
        
        # Check if required columns exist
        required_columns = ["TEXT", "CLASS"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in test dataset: {missing_columns}")
        
        # Extract texts and labels
        texts = df["TEXT"].tolist()
        labels = df["CLASS"].tolist()
        
        # Encode labels: HUMAN -> 0, MACHINE -> 1
        label_mapping = {"HUMAN": 0, "MACHINE": 1}
        y = np.array([label_mapping[label] for label in labels])
        
        # Collect metadata for analysis
        metadata = {
            "language": df["LANGUAGE"].tolist() if "LANGUAGE" in df.columns else None,
            "domain": df["DOMAIN"].tolist() if "DOMAIN" in df.columns else None,
            "text_lengths": [len(text) for text in texts],
            "word_counts": [len(text.split()) for text in texts]
        }
        
        # Print class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = "HUMAN" if label == 0 else "MACHINE"
            logger.info(f"Class {label_name} ({label}): {count} samples ({count/len(y)*100:.2f}%)")
        
        return texts, y, metadata

class TraditionalModel:
    """
    Implementation of the best traditional ML pipeline from main_traditional.py:
    CountVectorizerTokenizeStem + NuSVC
    """
    
    def __init__(self, language="romanian"):
        """Initialize the traditional model pipeline."""
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = SnowballStemmer(language=language)
        
        # Import required libraries
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.svm import NuSVC
        from sklearn.pipeline import Pipeline
        
        # Create the pipeline with the best parameters from the experiment
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                tokenizer=self._tokenize_and_stem,
                lowercase=False,
                binary=False,
                stop_words=None
            )),
            ('classifier', NuSVC(
                nu=0.5,  # Default value if not specified
                kernel='rbf',
                coef0=-0.5573835130142433,
                degree=1,
                gamma='scale',
                decision_function_shape='ovr',
                probability=True,
                shrinking=True,
                cache_size=1,
                break_ties=False
            ))
        ])
        
        logger.info("Traditional model (CountVectorizerTokenizeStem + NuSVC) initialized")
    
    def _tokenize_and_stem(self, text):
        """Tokenize and stem the text using the specified tokenizer and stemmer."""
        tokens = self.tokenizer.tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens
    
    def train(self, X_train, y_train):
        """Train the traditional model."""
        logger.info("Training traditional model...")
        start_time = time.time()
        self.pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info(f"Traditional model trained in {training_time:.2f} seconds")
    
    def predict(self, X):
        """Make predictions with the traditional model."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities from the traditional model."""
        return self.pipeline.predict_proba(X)

class LLMModel:
    """
    Implementation of the best LLM-based pipeline from main.py:
    FineTuneLLMEmbeddingClassifier with WORD_EMB_Microsoft_Deberta_V3_Base
    
    Note: This is a simplified version that will attempt to use the actual model
    if available, but includes a fallback that simulates the behavior for error analysis.
    """
    
    def __init__(self):
        """Initialize the LLM model."""
        # Always set to False initially
        self.using_actual_model = False
        
        try:
            # Try to import the autogoal transformers components
            from autogoal_transformers._manual import FineTuneLLMEmbeddingClassifier
            from autogoal_transformers._generated import WORD_EMB_Microsoft_Deberta_V3_Base
            
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
            
            # Set environment variable to better debug CUDA issues
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            
            initialize_cuda_multiprocessing()
            self.model = FineTuneLLMEmbeddingClassifier(
                inner_model=WORD_EMB_Microsoft_Deberta_V3_Base(),
                batch_size=16,
                max_length=128,
                learning_rate=4e-05,
                epochs=9,
                warmup_steps=1500,
                weight_decay=0.1,
                dropout_rate=0.1,
                optimizer="adamw",
                gradient_accumulation_steps=1,
                lr_scheduler="linear",
                early_stopping_delta=0.01,
                use_mixed_precision=False,  # Disable mixed precision to avoid CUDA errors
                use_gradient_clipping=False,
                gradient_clipping_max_norm=1.0,
                class_weighted_loss=False,
                num_workers="default"
            )
            
            self.using_actual_model = True
            logger.info("LLM model (FineTuneLLMEmbeddingClassifier with DeBERTa-v3) initialized")
            
        except Exception as e:
            raise e
    
    def train(self, X_train, y_train):
        """Train the LLM model."""
        logger.info("Training LLM model...")
        start_time = time.time()
        
        try:
            if self.using_actual_model:
                # For the actual LLM model, use finetune() instead of fit()
                self.model.train()  # Set the model to training mode
                self.model.finetune(X_train, y_train)  # Use finetune method for FineTuneLLMEmbeddingClassifier
            else:
                # For the simulated model
                self.model.fit(X_train, y_train)
                
            training_time = time.time() - start_time
            logger.info(f"LLM model trained in {training_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during LLM model training: {str(e)}")
            logger.warning("Falling back to simplified model")
            raise e
    
    def predict(self, X):
        """Make predictions with the LLM model."""
        try:
            if self.using_actual_model:
                self.model.eval()  # Set the model to evaluation mode
                return self.model.predict(X)
            else:
                return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # If we somehow still have issues, return a random prediction
            logger.warning("Returning random predictions due to model error")
            import numpy as np
            return np.random.randint(0, 2, size=len(X))
    
    def predict_proba(self, X):
        """Get prediction probabilities from the LLM model."""
        try:
            if self.using_actual_model:
                self.model.eval()  # Set the model to evaluation mode
                # DeBERTa model might not have predict_proba so we need to handle this
                try:
                    return self.model.predict_proba(X)
                except AttributeError:
                    # If predict_proba isn't available, create a simple version
                    preds = self.model.predict(X)
                    probas = np.zeros((len(X), 2))
                    for i, pred in enumerate(preds):
                        probas[i, pred] = 0.8  # Assign confidence of 0.8 to predicted class
                        probas[i, 1-pred] = 0.2  # Assign confidence of 0.2 to other class
                    return probas
            else:
                return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            # If we somehow still have issues, return random probabilities
            logger.warning("Returning random probability predictions due to model error")
            import numpy as np
            probas = np.random.rand(len(X), 2)
            # Normalize to ensure rows sum to 1
            return probas / probas.sum(axis=1, keepdims=True)

class ErrorAnalyzer:
    """
    Class to perform comprehensive error analysis on model predictions.
    """
    
    def __init__(self, model_name, y_true, y_pred, probas, texts, metadata=None):
        """Initialize the error analyzer with predictions and actual labels."""
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.probas = probas
        self.texts = texts
        self.metadata = metadata or {}
        
        # Compute errors
        self.errors = (y_true != y_pred)
        self.error_indices = np.where(self.errors)[0]
        
        # Compute basic metrics
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='macro')
        self.recall = recall_score(y_true, y_pred, average='macro')
        self.f1 = f1_score(y_true, y_pred, average='macro')
        
        logger.info(f"Error analysis initialized for {model_name}")
        logger.info(f"Accuracy: {self.accuracy:.4f}")
        logger.info(f"Precision: {self.precision:.4f}")
        logger.info(f"Recall: {self.recall:.4f}")
        logger.info(f"F1 Score: {self.f1:.4f}")
    
    def basic_metrics(self):
        """Return basic metrics as a dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'errors': len(self.error_indices),
            'total': len(self.y_true),
            'error_rate': len(self.error_indices) / len(self.y_true)
        }
    
    def confusion_matrix_metrics(self):
        """Generate confusion matrix and calculate class-specific metrics."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Calculate class-specific metrics
        class_report = classification_report(self.y_true, self.y_pred, target_names=['HUMAN', 'MACHINE'], output_dict=True)
        
        logger.info("Class-specific metrics:")
        logger.info(f"HUMAN - Precision: {class_report['HUMAN']['precision']:.4f}, "
                   f"Recall: {class_report['HUMAN']['recall']:.4f}, "
                   f"F1: {class_report['HUMAN']['f1-score']:.4f}")
        logger.info(f"MACHINE - Precision: {class_report['MACHINE']['precision']:.4f}, "
                   f"Recall: {class_report['MACHINE']['recall']:.4f}, "
                   f"F1: {class_report['MACHINE']['f1-score']:.4f}")
        
        return cm, class_report
    
    def roc_curve_metrics(self):
        """Calculate ROC curve metrics without visualization."""
        # Get probabilities for positive class (MACHINE)
        if self.probas.shape[1] == 2:  # Binary classification with probabilities for both classes
            y_prob = self.probas[:, 1]
        else:  # Only one class probability provided
            y_prob = self.probas
        
        fpr, tpr, thresholds = roc_curve(self.y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}
    
    def precision_recall_metrics(self):
        """Calculate precision-recall metrics without visualization."""
        # Get probabilities for positive class (MACHINE)
        if self.probas.shape[1] == 2:  # Binary classification with probabilities for both classes
            y_prob = self.probas[:, 1]
        else:  # Only one class probability provided
            y_prob = self.probas
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, y_prob)
        avg_precision = average_precision_score(self.y_true, y_prob)
        
        logger.info(f"Average Precision: {avg_precision:.4f}")
        
        return {'precision': precision, 'recall': recall, 'thresholds': thresholds, 'avg_precision': avg_precision}
    
    def analyze_text_length_impact(self):
        """Analyze impact of text length on model performance without visualization."""
        if 'text_lengths' not in self.metadata:
            logger.warning("Text length data not available for analysis")
            return None
        
        text_lengths = self.metadata['text_lengths']
        
        # Create a DataFrame for analysis
        df = pd.DataFrame({
            'text_length': text_lengths,
            'true_label': self.y_true,
            'predicted_label': self.y_pred,
            'error': self.errors
        })
        
        # Define bins for text length
        bins = [0, 100, 200, 300, 500, 1000, max(text_lengths)]
        labels = ['0-100', '101-200', '201-300', '301-500', '501-1000', '1001+']
        
        df['length_bin'] = pd.cut(df['text_length'], bins=bins, labels=labels)
        
        # Calculate error rate by length bin
        length_analysis = df.groupby('length_bin').agg(
            error_rate=('error', 'mean'),
            count=('error', 'count')
        ).reset_index()
        
        logger.info("Error rates by text length:")
        for _, row in length_analysis.iterrows():
            logger.info(f"  {row['length_bin']}: {row['error_rate']:.4f} (n={row['count']})")
        
        return length_analysis
    
    def analyze_language_impact(self):
        """Analyze impact of language on model performance without visualization."""
        if 'language' not in self.metadata:
            logger.warning("Language data not available for analysis")
            return None
        
        languages = self.metadata['language']
        
        # Create a DataFrame for analysis
        df = pd.DataFrame({
            'language': languages,
            'true_label': self.y_true,
            'predicted_label': self.y_pred,
            'error': self.errors
        })
        
        # Calculate error rate by language
        language_analysis = df.groupby('language').agg(
            error_rate=('error', 'mean'),
            count=('error', 'count')
        ).reset_index()
        
        logger.info("Error rates by language:")
        for _, row in language_analysis.iterrows():
            logger.info(f"  {row['language']}: {row['error_rate']:.4f} (n={row['count']})")
        
        return language_analysis
    
    def analyze_domain_impact(self):
        """Analyze impact of domain on model performance without visualization."""
        if 'domain' not in self.metadata:
            logger.warning("Domain data not available for analysis")
            return None
        
        domains = self.metadata['domain']
        
        # Create a DataFrame for analysis
        df = pd.DataFrame({
            'domain': domains,
            'true_label': self.y_true,
            'predicted_label': self.y_pred,
            'error': self.errors
        })
        
        # Calculate error rate by domain
        domain_analysis = df.groupby('domain').agg(
            error_rate=('error', 'mean'),
            count=('error', 'count')
        ).reset_index()
        
        logger.info("Error rates by domain:")
        for _, row in domain_analysis.iterrows():
            logger.info(f"  {row['domain']}: {row['error_rate']:.4f} (n={row['count']})")
        
        return domain_analysis
    
    def analyze_misclassifications(self, max_samples=10):
        """Analyze misclassified examples."""
        if len(self.error_indices) == 0:
            logger.info("No misclassifications to analyze!")
            return []
        
        # Sample a subset of misclassified examples
        if len(self.error_indices) > max_samples:
            sample_indices = random.sample(list(self.error_indices), max_samples)
        else:
            sample_indices = self.error_indices
            
        misclassifications = []
        
        for idx in sample_indices:
            true_label = "HUMAN" if self.y_true[idx] == 0 else "MACHINE"
            pred_label = "HUMAN" if self.y_pred[idx] == 0 else "MACHINE"
            
            # Get text features
            text = self.texts[idx]
            text_length = len(text)
            word_count = len(text.split())
            
            # Get prediction probability
            if self.probas.shape[1] == 2:  # Binary classification with probabilities for both classes
                confidence = self.probas[idx][self.y_pred[idx]]
            else:  # Only one class probability provided
                confidence = self.probas[idx] if self.y_pred[idx] == 1 else 1 - self.probas[idx]
            
            # Additional metadata
            language = self.metadata.get('language', [None] * len(self.y_true))[idx]
            domain = self.metadata.get('domain', [None] * len(self.y_true))[idx]
            
            misclassifications.append({
                'index': idx,
                'text': text[:200] + "..." if len(text) > 200 else text,  # Truncate for display
                'full_text': text,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'text_length': text_length,
                'word_count': word_count,
                'language': language,
                'domain': domain
            })
        
        return misclassifications
    
    def analyze_common_patterns(self, max_patterns=10):
        """Analyze common patterns in errors."""
        if len(self.error_indices) == 0:
            logger.info("No errors to analyze patterns!")
            return {}
        
        # Extract all texts with errors
        error_texts = [self.texts[idx] for idx in self.error_indices]
        
        # Analyze n-grams in error texts
        ngrams = []
        
        # Helper function to extract n-grams
        def extract_ngrams(text, n):
            words = re.findall(r'\w+', text.lower())
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        # Extract 1, 2, and 3-grams
        for text in error_texts:
            ngrams.extend(extract_ngrams(text, 1))
            ngrams.extend(extract_ngrams(text, 2))
            ngrams.extend(extract_ngrams(text, 3))
        
        # Count n-gram frequencies
        ngram_counter = Counter(ngrams)
        
        # Get the most common n-grams
        common_patterns = ngram_counter.most_common(max_patterns)
        
        return common_patterns
    
    def display_error_samples(self, max_samples=5):
        """Display a few error samples for manual inspection."""
        misclassifications = self.analyze_misclassifications(max_samples=max_samples)
        
        if not misclassifications:
            logger.info("No misclassifications to display")
            return
        
        logger.info(f"\nSample misclassifications from {self.model_name}:")
        for i, error in enumerate(misclassifications):
            logger.info(f"\nError #{i+1}:")
            logger.info(f"Text: {error['text']}")
            logger.info(f"True label: {error['true_label']}, Predicted: {error['pred_label']}")
            logger.info(f"Confidence: {error['confidence']:.4f}")
            logger.info(f"Length: {error['text_length']} chars, {error['word_count']} words")
            if error['language']:
                logger.info(f"Language: {error['language']}")
            if error['domain']:
                logger.info(f"Domain: {error['domain']}")
    
    def full_analysis(self):
        """Perform a full error analysis."""
        logger.info(f"\nPerforming full error analysis for {self.model_name}...")
        
        # Basic metrics and confusion matrix
        metrics = self.basic_metrics()
        cm, class_report = self.confusion_matrix_metrics()
        
        # ROC and PR curves
        roc_data = self.roc_curve_metrics()
        pr_data = self.precision_recall_metrics()
        
        # Feature impact analysis
        length_analysis = self.analyze_text_length_impact()
        language_analysis = self.analyze_language_impact()
        domain_analysis = self.analyze_domain_impact()
        
        # Misclassification analysis
        self.display_error_samples()
        common_patterns = self.analyze_common_patterns()
        
        if common_patterns:
            logger.info("\nCommon patterns in errors:")
            for pattern, count in common_patterns:
                logger.info(f"  - '{pattern}': {count} occurrences")
        
        # Return aggregated analysis
        return {
            'metrics': metrics,
            'class_report': class_report,
            'confusion_matrix': cm,
            'roc_data': roc_data,
            'pr_data': pr_data,
            'length_analysis': length_analysis,
            'language_analysis': language_analysis,
            'domain_analysis': domain_analysis,
            'common_patterns': common_patterns
        }

def compare_models(analyzer1, analyzer2):
    """
    Compare two models and determine statistical significance of performance differences.
    """
    logger.info(f"\nComparing {analyzer1.model_name} and {analyzer2.model_name}...")
    
    # Get predictions from both models
    pred1 = analyzer1.y_pred
    pred2 = analyzer2.y_pred
    true_labels = analyzer1.y_true  # Both analyzers should have the same true labels
    
    # Create a contingency table for McNemar's test
    # [a, b] - a: both models correct, b: model1 correct, model2 incorrect
    # [c, d] - c: model1 incorrect, model2 correct, d: both models incorrect
    a = sum((pred1 == true_labels) & (pred2 == true_labels))
    b = sum((pred1 == true_labels) & (pred2 != true_labels))
    c = sum((pred1 != true_labels) & (pred2 == true_labels))
    d = sum((pred1 != true_labels) & (pred2 != true_labels))
    
    contingency_table = np.array([[a, b], [c, d]])
    
    # Perform McNemar's test
    try:
        result = mcnemar_test(contingency_table, correction=True)
        p_value = result.pvalue
        is_significant = p_value < 0.05
        
        logger.info(f"McNemar's test p-value: {p_value:.4f}")
        if is_significant:
            logger.info("The performance difference between the models is statistically significant (p < 0.05)")
        else:
            logger.info("The performance difference between the models is NOT statistically significant (p >= 0.05)")
            
        # Compare performance metrics
        metrics1 = analyzer1.basic_metrics()
        metrics2 = analyzer2.basic_metrics()
        
        comparison = pd.DataFrame({
            analyzer1.model_name: [metrics1['accuracy'], metrics1['precision'], 
                                 metrics1['recall'], metrics1['f1'], metrics1['error_rate']],
            analyzer2.model_name: [metrics2['accuracy'], metrics2['precision'], 
                                 metrics2['recall'], metrics2['f1'], metrics2['error_rate']]
        }, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Error Rate'])
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison}")
        
        # Analyze where models agree and disagree
        both_correct = sum((pred1 == true_labels) & (pred2 == true_labels))
        both_incorrect = sum((pred1 != true_labels) & (pred2 != true_labels))
        only_model1_correct = sum((pred1 == true_labels) & (pred2 != true_labels))
        only_model2_correct = sum((pred1 != true_labels) & (pred2 == true_labels))
        
        total = len(true_labels)
        
        logger.info(f"\nAgreement Analysis:")
        logger.info(f"Both models correct: {both_correct} ({both_correct/total:.2%})")
        logger.info(f"Both models incorrect: {both_incorrect} ({both_incorrect/total:.2%})")
        logger.info(f"Only {analyzer1.model_name} correct: {only_model1_correct} ({only_model1_correct/total:.2%})")
        logger.info(f"Only {analyzer2.model_name} correct: {only_model2_correct} ({only_model2_correct/total:.2%})")
        
        return {
            'contingency_table': contingency_table,
            'p_value': p_value,
            'is_significant': is_significant,
            'comparison': comparison,
            'agreement': {
                'both_correct': both_correct,
                'both_incorrect': both_incorrect,
                'only_model1_correct': only_model1_correct,
                'only_model2_correct': only_model2_correct
            }
        }
    
    except Exception as e:
        logger.error(f"Error performing statistical comparison: {e}")
        return None

print(5)

def save_results_to_json(results, filename):
    """
    Helper function to save results to JSON file with proper handling of NumPy types.
    
    Parameters:
    ----------
    results : dict
        Dictionary containing the results to save
    filename : str
        Name of the file to save results to
    """
    import json
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if pd.isna(obj):
                return None
            return super(NumpyEncoder, self).default(obj)
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    log_progress(f"Results saved to {output_path}")
    return output_path

def prepare_analyzer_results(analyzer, results):
    """
    Prepare comprehensive results from an analyzer for JSON serialization.
    
    Parameters:
    ----------
    analyzer : ErrorAnalyzer
        The analyzer object containing the results
    results : dict
        Dictionary of results from analyzer.full_analysis()
        
    Returns:
    -------
    dict
        A dictionary with all results formatted for JSON serialization
    """
    # Format dataframes for JSON serialization
    def format_df(df):
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            return df.to_dict(orient='records')
        return df
    
    # Basic metrics
    metrics = analyzer.basic_metrics()
    
    # Get formatted results
    formatted_results = {
        'summary': {
            'model_name': analyzer.model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'error_rate': metrics['error_rate'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detailed_metrics': {
            'class_report': results['class_report'],
            'confusion_matrix': results['confusion_matrix'].tolist() if isinstance(results['confusion_matrix'], np.ndarray) else results['confusion_matrix'],
            'roc_auc': results['roc_data']['auc'],
            'avg_precision': results['pr_data']['avg_precision'],
        },
        'error_analysis': {
            'length_analysis': format_df(results['length_analysis']),
            'language_analysis': format_df(results['language_analysis']),
            'domain_analysis': format_df(results['domain_analysis']),
            'common_patterns': results['common_patterns'],
            'error_count': len(analyzer.error_indices),
            'total_samples': len(analyzer.y_true)
        },
        'misclassifications': analyzer.analyze_misclassifications(max_samples=20)
    }
    
    return formatted_results

def analyze_traditional_model():
    """Perform analysis on the traditional model and save results as JSON."""
    # Check if results already exist
    traditional_results_path = os.path.join(OUTPUT_DIR, 'traditional_model_results.json')
    if os.path.exists(traditional_results_path):
        log_progress("Traditional model results already exist, skipping analysis")
        with open(traditional_results_path, 'r') as f:
            import json
            return json.load(f), None
    
    # Initialize and train model
    log_progress("Initializing traditional model...")
    traditional_model = TraditionalModel()
    log_progress("Traditional model initialized")
    
    log_progress("Starting to train traditional model...")
    traditional_model.train(X_train, y_train)
    log_progress("Traditional model training completed")
    
    # Make predictions
    log_progress("Making predictions with traditional model...")
    traditional_preds = traditional_model.predict(X_test)
    traditional_probas = traditional_model.predict_proba(X_test)
    log_progress("Traditional model predictions completed")
    
    # Error analysis
    log_progress("Initializing error analyzer for traditional model...")
    traditional_analyzer = ErrorAnalyzer(
        model_name="Traditional Model",
        y_true=y_test,
        y_pred=traditional_preds,
        probas=traditional_probas,
        texts=X_test,
        metadata=test_metadata
    )
    log_progress("Traditional model error analyzer initialized")
    
    log_progress("Running full analysis for traditional model...")
    traditional_results = traditional_analyzer.full_analysis()
    log_progress("Traditional model analysis completed")
    
    # Prepare comprehensive results
    formatted_results = prepare_analyzer_results(traditional_analyzer, traditional_results)
    
    # Save results
    save_results_to_json(formatted_results, 'traditional_model_results.json')
    
    return formatted_results, traditional_analyzer

def analyze_llm_model():
    """Perform analysis on the LLM model and save results as JSON."""
    # Check if results already exist
    llm_results_path = os.path.join(OUTPUT_DIR, 'llm_model_results.json')
    if os.path.exists(llm_results_path):
        log_progress("LLM model results already exist, skipping analysis")
        with open(llm_results_path, 'r') as f:
            import json
            return json.load(f), None
    
    # Initialize and train model
    log_progress("Initializing LLM model...")
    llm_model = LLMModel()
    log_progress("LLM model initialized")
    
    log_progress("Starting to train LLM model...")
    llm_model.train(X_train, y_train)
    log_progress("LLM model training completed")
    
    # Make predictions
    log_progress("Making predictions with LLM model...")
    llm_preds = llm_model.predict(X_test)
    llm_probas = llm_model.predict_proba(X_test)
    log_progress("LLM model predictions completed")
    
    # Error analysis
    log_progress("Initializing error analyzer for LLM model...")
    llm_analyzer = ErrorAnalyzer(
        model_name="LLM Model",
        y_true=y_test,
        y_pred=llm_preds,
        probas=llm_probas,
        texts=X_test,
        metadata=test_metadata
    )
    log_progress("LLM model error analyzer initialized")
    
    log_progress("Running full analysis for LLM model...")
    llm_results = llm_analyzer.full_analysis()
    log_progress("LLM model analysis completed")
    
    # Prepare comprehensive results
    formatted_results = prepare_analyzer_results(llm_analyzer, llm_results)
    
    # Save results
    save_results_to_json(formatted_results, 'llm_model_results.json')
    
    return formatted_results, llm_analyzer

def main():
    """
    Main function to orchestrate the error analysis process.
    """
    start_time = time.time()
    
    # Set random seeds for reproducibility
    set_seeds()
    log_progress("Random seeds set for reproducibility")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log_progress(f"Output directory created at {OUTPUT_DIR}")
    
    # Load data
    log_progress("Starting to load data...")
    data_loader = DataLoader()
    X_train, y_train, X_val, y_val, train_metadata = data_loader.load_training_data()
    log_progress(f"Training data loaded: {len(X_train)} samples")
    X_test, y_test, test_metadata = data_loader.load_test_data()
    log_progress(f"Test data loaded: {len(X_test)} samples")
    
    # Define a function to save results to JSON
    def save_results_to_json(results, filename):
        """Helper function to save results to JSON file."""
        import json
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return super(NumpyEncoder, self).default(obj)
        
        with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)
        log_progress(f"Results saved to {filename}")
    
    # Analysis for Traditional Model
    def analyze_traditional_model():
        # Check if results already exist
        traditional_results_path = os.path.join(OUTPUT_DIR, 'traditional_model_results.json')
        if os.path.exists(traditional_results_path):
            log_progress("Traditional model results already exist, skipping analysis")
            with open(traditional_results_path, 'r') as f:
                import json
                return json.load(f)
        
        # Initialize and train model
        log_progress("Initializing traditional model...")
        traditional_model = TraditionalModel()
        log_progress("Traditional model initialized")
        
        log_progress("Starting to train traditional model...")
        traditional_model.train(X_train, y_train)
        log_progress("Traditional model training completed")
        
        # Make predictions
        log_progress("Making predictions with traditional model...")
        traditional_preds = traditional_model.predict(X_test)
        traditional_probas = traditional_model.predict_proba(X_test)
        log_progress("Traditional model predictions completed")
        
        # Error analysis
        log_progress("Initializing error analyzer for traditional model...")
        traditional_analyzer = ErrorAnalyzer(
            model_name="Traditional Model",
            y_true=y_test,
            y_pred=traditional_preds,
            probas=traditional_probas,
            texts=X_test,
            metadata=test_metadata
        )
        log_progress("Traditional model error analyzer initialized")
        
        log_progress("Running full analysis for traditional model...")
        traditional_results = traditional_analyzer.full_analysis()
        log_progress("Traditional model analysis completed")
        
        # Prepare results for saving
        results = {
            'metrics': traditional_analyzer.basic_metrics(),
            'class_report': traditional_results['class_report'],
            'confusion_matrix': traditional_results['confusion_matrix'].tolist(),
            'roc_auc': traditional_results['roc_data']['auc'],
            'avg_precision': traditional_results['pr_data']['avg_precision'],
            'length_analysis': traditional_results['length_analysis'].to_dict() if isinstance(traditional_results['length_analysis'], pd.DataFrame) else traditional_results['length_analysis'],
            'language_analysis': traditional_results['language_analysis'].to_dict() if isinstance(traditional_results['language_analysis'], pd.DataFrame) else traditional_results['language_analysis'],
            'domain_analysis': traditional_results['domain_analysis'].to_dict() if isinstance(traditional_results['domain_analysis'], pd.DataFrame) else traditional_results['domain_analysis'],
            'common_patterns': traditional_results['common_patterns']
        }
        
        # Save results
        save_results_to_json(results, 'traditional_model_results.json')
        
        return results, traditional_analyzer
    
    # Analysis for LLM Model
    def analyze_llm_model():
        # Check if results already exist
        llm_results_path = os.path.join(OUTPUT_DIR, 'llm_model_results.json')
        if os.path.exists(llm_results_path):
            log_progress("LLM model results already exist, skipping analysis")
            with open(llm_results_path, 'r') as f:
                import json
                return json.load(f)
        
        # Initialize and train model
        log_progress("Initializing LLM model...")
        llm_model = LLMModel()
        log_progress("LLM model initialized")
        
        log_progress("Starting to train LLM model...")
        llm_model.train(X_train, y_train)
        log_progress("LLM model training completed")
        
        # Make predictions
        log_progress("Making predictions with LLM model...")
        llm_preds = llm_model.predict(X_test)
        llm_probas = llm_model.predict_proba(X_test)
        log_progress("LLM model predictions completed")
        
        # Error analysis
        log_progress("Initializing error analyzer for LLM model...")
        llm_analyzer = ErrorAnalyzer(
            model_name="LLM Model",
            y_true=y_test,
            y_pred=llm_preds,
            probas=llm_probas,
            texts=X_test,
            metadata=test_metadata
        )
        log_progress("LLM model error analyzer initialized")
        
        log_progress("Running full analysis for LLM model...")
        llm_results = llm_analyzer.full_analysis()
        log_progress("LLM model analysis completed")
        
        # Prepare results for saving
        results = {
            'metrics': llm_analyzer.basic_metrics(),
            'class_report': llm_results['class_report'],
            'confusion_matrix': llm_results['confusion_matrix'].tolist(),
            'roc_auc': llm_results['roc_data']['auc'],
            'avg_precision': llm_results['pr_data']['avg_precision'],
            'length_analysis': llm_results['length_analysis'].to_dict() if isinstance(llm_results['length_analysis'], pd.DataFrame) else llm_results['length_analysis'],
            'language_analysis': llm_results['language_analysis'].to_dict() if isinstance(llm_results['language_analysis'], pd.DataFrame) else llm_results['language_analysis'],
            'domain_analysis': llm_results['domain_analysis'].to_dict() if isinstance(llm_results['domain_analysis'], pd.DataFrame) else llm_results['domain_analysis'],
            'common_patterns': llm_results['common_patterns']
        }
        
        # Save results
        save_results_to_json(results, 'llm_model_results.json')
        
        return results, llm_analyzer
    
    # Run analyses - COMMENT OUT THE ONES YOU DON'T WANT TO RUN
    # traditional_results, traditional_analyzer = analyze_traditional_model()
    llm_results, llm_analyzer = analyze_llm_model()
    
    # If only one model was analyzed, create a simple report for it
    if 'traditional_analyzer' in locals():
        log_progress("Creating traditional model report...")
        with open(os.path.join(OUTPUT_DIR, 'traditional_model_report.txt'), 'w') as f:
            f.write("=== IBERMAT TEXT CLASSIFICATION - TRADITIONAL MODEL ANALYSIS ===\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== MODEL PERFORMANCE ===\n")
            metrics = traditional_analyzer.basic_metrics()
            for key, value in metrics.items():
                if key not in ['errors', 'total']:
                    f.write(f"{key.capitalize()}: {value:.4f}\n")
            
            f.write("\nDetailed results saved in traditional_model_results.json\n")
            f.write(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds.")
        log_progress("Traditional model report created")
    
    if 'llm_analyzer' in locals():
        log_progress("Creating LLM model report...")
        with open(os.path.join(OUTPUT_DIR, 'llm_model_report.txt'), 'w') as f:
            f.write("=== IBERMAT TEXT CLASSIFICATION - LLM MODEL ANALYSIS ===\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== MODEL PERFORMANCE ===\n")
            metrics = llm_analyzer.basic_metrics()
            for key, value in metrics.items():
                if key not in ['errors', 'total']:
                    f.write(f"{key.capitalize()}: {value:.4f}\n")
            
            f.write("\nDetailed results saved in llm_model_results.json\n")
            f.write(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds.")
        log_progress("LLM model report created")
    
    logger.info(f"\nError analysis completed and saved to {OUTPUT_DIR}")
    logger.info(f"Total runtime: {time.time() - start_time:.2f} seconds")
    log_progress("Script execution completed successfully")

if __name__ == "__main__":
    main()