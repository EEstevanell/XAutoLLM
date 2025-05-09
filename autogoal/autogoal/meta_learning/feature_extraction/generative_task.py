from autogoal.meta_learning.feature_extraction._base import FeatureExtractor
import numpy as np # Standard import
import string # Standard import

import torch # Standard import
import nltk # Standard import
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer # Standard import
from sklearn.metrics.pairwise import cosine_similarity # Standard import
from rouge_score import rouge_scorer # Standard import
from tqdm.auto import tqdm # For progress bars

class GenerativeTaskFeatureExtractor(FeatureExtractor):
    NUM_REGULAR_FEATURES = 11 # Constant for the number of regular features

    def __init__(self, batch_size_sbert=128): # Allow configuring batch_size
        self.batch_size_sbert = batch_size_sbert
        try:
            from nltk.corpus import stopwords
            # NLTK resource check
            try:
                stopwords.words('english')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except ImportError as e:
            # This specific handling is for NLTK not being installed at all.
            # The LookupError for missing resources is handled above.
            # Consider if this block is reachable if NLTK itself fails to import earlier.
            # Typically, the top-level import numpy as np etc. would fail first.
            # For safety, we can keep it, or rely on top-level import failures.
            raise ImportError("NLTK not installed or resources missing. Please install NLTK and run nltk.download('stopwords') and nltk.download('punkt').") from e


        self._stopwords = set(stopwords.words('english'))
        self._string_punctuation = string.punctuation # Store punctuation string

        # Initialize SentenceTransformer model
        try:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = SentenceTransformer('all-MiniLM-L6-v2', device=self._device)
            self.EMBEDDING_DIM = self._model.get_sentence_embedding_dimension()
        except Exception as e: # Catch potential errors during model loading
            raise ImportError(f"SentenceTransformers model 'all-MiniLM-L6-v2' could not be loaded. Ensure library is installed and model is accessible. Error: {e}")


        # Initialize ROUGE scorer instance
        try:
            self.rouge_l_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except Exception as e: # Catch potential errors during ROUGE scorer init
            raise ImportError(f"RougeScorer could not be initialized. Ensure rouge-score library is installed correctly. Error: {e}")


    def _normalize_and_tokenize(self, text):
        text = text.lower()
        # Use stored string.punctuation
        text = text.translate(str.maketrans('', '', self._string_punctuation))
        tokens = word_tokenize(text) # Directly use imported word_tokenize
        tokens = [t for t in tokens if t.isalpha() and t not in self._stopwords]
        return tokens

    def extract_features(self, X_train, y_train):
        if not X_train or not y_train or len(X_train) != len(y_train):
            # Return NaNs for regular, zeros for semantic
            regular = np.full(self.NUM_REGULAR_FEATURES, np.nan)
            semantic = np.zeros(self.EMBEDDING_DIM)
            return {"meta": regular.tolist(), "semantic": semantic.tolist()} # Updated return format

        # 1. Number of samples
        n_samples = float(len(X_train))

        # 2-3. Prompt char length stats
        prompt_lens_char = np.array([len(x) for x in X_train])
        avg_prompt_len_char = np.mean(prompt_lens_char)
        std_prompt_len_char = np.std(prompt_lens_char)

        # 4. Prompt lexical diversity (TTR)
        # For TTR, process tokens per document then aggregate, or concatenate then tokenize.
        # Concatenating first is simpler as implemented.
        all_prompts_text = ' '.join(X_train)
        prompt_tokens_for_ttr = self._normalize_and_tokenize(all_prompts_text)
        prompt_ttr = len(set(prompt_tokens_for_ttr)) / (len(prompt_tokens_for_ttr) + 1e-9) if prompt_tokens_for_ttr else 0.0

        # 5-6. Target char length stats
        target_lens_char = np.array([len(y) for y in y_train])
        avg_target_len_char = np.mean(target_lens_char)
        std_target_len_char = np.std(target_lens_char)

        # 7. Target lexical diversity (TTR)
        all_targets_text = ' '.join(y_train)
        target_tokens_for_ttr = self._normalize_and_tokenize(all_targets_text)
        target_ttr = len(set(target_tokens_for_ttr)) / (len(target_tokens_for_ttr) + 1e-9) if target_tokens_for_ttr else 0.0
        
        # 8. Avg char length ratio (target/prompt)
        ratios_char = [len(y) / (len(x) + 1e-9) if len(x) > 0 else 0.0 for x, y in zip(X_train, y_train)]
        avg_ratio_char = np.mean(ratios_char)

        # 9. Vocabulary novelty (target vs prompt)
        vocab_novelty_scores = []
        # Using tqdm for progress on this loop
        for x, y in tqdm(zip(X_train, y_train), total=len(X_train), desc="Calculating Vocab Novelty"):
            prompt_token_set = set(self._normalize_and_tokenize(x))
            target_token_list = self._normalize_and_tokenize(y)
            if target_token_list:
                novel_count = sum(1 for t in target_token_list if t not in prompt_token_set)
                novelty = novel_count / (len(target_token_list) + 1e-9)
            else:
                novelty = 0.0
            vocab_novelty_scores.append(novelty)
        avg_vocab_novelty = np.mean(vocab_novelty_scores) if vocab_novelty_scores else 0.0

        # 10. Avg prompt-target pairwise semantic similarity
        # Use show_progress_bar for SentenceTransformer's encode method
        # No need to pass device= again as model is already on the device
        embeddings_prompt = self._model.encode(
            X_train,
            batch_size=self.batch_size_sbert,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings_target = self._model.encode(
            y_train,
            batch_size=self.batch_size_sbert,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Reshape for cosine_similarity if it expects 2D arrays for pairwise comparison
        # The current loop is fine as it compares one pair at a time.
        similarities = [cosine_similarity(ep.reshape(1, -1), et.reshape(1, -1))[0,0] 
                        for ep, et in zip(embeddings_prompt, embeddings_target)]
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # 11. Avg ROUGE-L F1 (prompt vs target)
        # Use the instance from __init__
        rouge_f1s = [self.rouge_l_scorer_instance.score(x, y)['rougeL'].fmeasure 
                       for x, y in tqdm(zip(X_train, y_train), total=len(X_train), desc="Calculating ROUGE-L F1")]
        avg_rouge_f1 = np.mean(rouge_f1s) if rouge_f1s else 0.0

        regular_feature_vector = np.array([
            n_samples,
            avg_prompt_len_char,
            std_prompt_len_char,
            prompt_ttr,
            avg_target_len_char,
            std_target_len_char,
            target_ttr,
            avg_ratio_char,
            avg_vocab_novelty,
            avg_similarity,
            avg_rouge_f1
        ], dtype=float)

        # Semantic feature: mean prompt embedding
        mean_prompt_embedding = np.mean(embeddings_prompt, axis=0) if embeddings_prompt.size > 0 else np.zeros(self.EMBEDDING_DIM)

        return {"meta": regular_feature_vector, "semantic": mean_prompt_embedding} # Return np.ndarray directly

