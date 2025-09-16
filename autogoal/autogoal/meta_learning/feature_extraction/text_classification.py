from autogoal.meta_learning.feature_extraction._base import FeatureExtractor

class TextClassificationFeatureExtractor(FeatureExtractor):
    def extract_features(self, X_train, y_train):
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy not installed. Please install via 'pip install numpy'.") from e
        
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
        except ImportError as e:
            raise ImportError("Scikit-learn not installed. Please install via 'pip install scikit-learn'.") from e
        
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                # Check if stopwords are available
                stopwords.words('english')
            except LookupError:
                # Download stopwords if not already available
                nltk.download('stopwords', quiet=True)
        except ImportError as e:
            raise ImportError("NLTK not installed. Please install via 'pip install nltk'.") from e
        
        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=object) # Ensure it handles strings if X_train is list of strings
        y_train = np.array(y_train)
        
        # Compute training set metrics
        n_instances = int(len(X_train))
        n_classes = int(len(np.unique(y_train)))
        
        class_counts = np.array([np.sum(y_train == cls) for cls in np.unique(y_train)])
        class_probs = class_counts / n_instances if n_instances > 0 else np.zeros_like(class_counts, dtype=float)
        
        # Handle cases where a class might have zero probability to avoid log2(0)
        valid_class_probs = class_probs[class_probs > 0]
        if len(valid_class_probs) > 0:
            class_entropy = -np.sum(valid_class_probs * np.log2(valid_class_probs))
        else:
            class_entropy = 0.0

        min_class_prob = np.min(class_probs) if len(class_probs) > 0 else 0.0
        max_class_prob = np.max(class_probs) if len(class_probs) > 0 else 0.0
        imbalance_ratio = min_class_prob / max_class_prob if max_class_prob > 0 else 0.0
        
        # Document lengths (in characters) for training set
        # Ensure X_train elements are strings before calling len()
        doc_lengths = np.array([len(str(doc)) for doc in X_train])
        avg_doc_length = np.mean(doc_lengths) if len(doc_lengths) > 0 else 0.0
        std_doc_length = np.std(doc_lengths) if len(doc_lengths) > 0 else 0.0
        coef_var_doc_length = std_doc_length / avg_doc_length if avg_doc_length != 0 else 0.0
        
        decision_tree_accuracy = 0.0 # Default value
        if n_instances > 0 and X_train.size > 0 : # Proceed only if there's data
            try:
                # Landmarker: Decision Tree accuracy using vectorized text features
                # Vectorize text data
                # Ensure X_train is 1D array of strings for TfidfVectorizer
                if X_train.ndim > 1:
                    # If X_train is not 1D, attempt to flatten or take first element,
                    # This depends on expected structure. Assuming it should be 1D list of texts.
                    # For now, if it's not, we might skip this landmarker or raise error.
                    # Let's assume it's a list of strings as per typical text classification.
                    pass # Already ensured X_train is np.array(X_train, dtype=object)

                vectorizer = TfidfVectorizer(max_features=min(5000, n_instances // 2 if n_instances > 1 else 1), 
                                             stop_words=set(stopwords.words('english')))
                X_vectorized = vectorizer.fit_transform(X_train.astype('U')) # astype('U') for unicode strings
                
                # Dimensionality reduction to reduce computational load
                # n_components for SVD must be < n_features (X_vectorized.shape[1])
                # and also < n_samples (X_vectorized.shape[0])
                n_svd_components = min(100, X_vectorized.shape[1] -1, X_vectorized.shape[0] -1 )
                if n_svd_components > 0 :
                    svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
                    X_reduced = svd.fit_transform(X_vectorized)
                    
                    # Train decision tree classifier with stratified K-fold cross-validation
                    # n_splits for StratifiedKFold must be <= n_samples_in_smallest_class
                    min_samples_in_class = np.min(class_counts) if len(class_counts) > 0 else 0
                    n_cv_splits = min(5, min_samples_in_class)

                    if n_cv_splits >= 2 and X_reduced.shape[0] >= n_cv_splits: # Ensure enough samples for CV
                        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
                        skf = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(clf, X_reduced, y_train, cv=skf)
                        decision_tree_accuracy = np.mean(cv_scores)
                    else:
                        # Not enough samples for meaningful CV, or only one class
                        # Fallback: train on all data and evaluate on itself (not ideal, but a landmarker)
                        if X_reduced.shape[0] > 0:
                             clf = DecisionTreeClassifier(max_depth=5, random_state=42)
                             clf.fit(X_reduced, y_train)
                             decision_tree_accuracy = clf.score(X_reduced, y_train)

            except Exception as e:
                # print(f"Warning: Could not compute decision tree landmarker: {e}")
                decision_tree_accuracy = 0.0 # Default on error
        
        # Combine all features into the feature vector
        meta_feature_vector = np.array([
            n_instances,
            n_classes,
            class_entropy,
            min_class_prob,
            max_class_prob,
            imbalance_ratio,
            avg_doc_length,
            std_doc_length,
            coef_var_doc_length,
            decision_tree_accuracy,
        ], dtype=float) # Ensure float type
        
        # For now, no semantic features are computed by this extractor
        semantic_feature_vector = None 
        
        return {
            "meta": meta_feature_vector, # Return np.ndarray directly
            "semantic": semantic_feature_vector # This is already None or would be np.ndarray
        }