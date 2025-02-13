from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn_crfsuite import CRF

from autogoal_sklearn._builder import SklearnTransformer, SklearnEstimator
from autogoal.kb import *
from autogoal.grammar import (
    BooleanValue,
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
)
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase, Supervised
import numpy as np
from scipy.sparse import vstack


from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures

class PolynomialFeatures(_PolynomialFeatures, SklearnTransformer):
    def __init__(
        self, 
        degree: DiscreteValue(1, 4)
        ):
        SklearnTransformer.__init__(self)
        _PolynomialFeatures.__init__(self, degree=degree)

    def run(self, input: MatrixContinuousDense) -> MatrixContinuousDense:
        return SklearnTransformer.run(self, input)

@nice_repr
class CountVectorizerStopwordTokenizeStem(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: BooleanValue(),
        stopwords_remove: BooleanValue(),
        binary: BooleanValue(),
        inner_tokenizer: algorithm(Sentence, Seq[Word]),
        inner_stemmer: algorithm(Word, Stem),
        inner_stopwords: algorithm(Seq[Word], Seq[Word]),
    ):
        self.stopwords_remove = stopwords_remove
        self.inner_tokenizer = inner_tokenizer
        self.inner_stemmer = inner_stemmer
        self.inner_stopwords = inner_stopwords

        SklearnTransformer.__init__(self)
        _CountVectorizer.__init__(self, lowercase=lowercase, binary=binary)

    def build_tokenizer(self):
        def func(sentence):
            tokens = self.inner_tokenizer.run(sentence)
            tokens = (
                self.inner_stopwords.run(sentence) if self.stopwords_remove else tokens
            )
            return [self.inner_stemmer.run(token) for token in tokens]

        return func

    def run(self, input: Seq[Sentence]) -> MatrixContinuousSparse:
        return SklearnTransformer.run(self, input)

@nice_repr
class CountVectorizerTokenizeStem(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: BooleanValue(),
        stopwords_remove: BooleanValue(),
        binary: BooleanValue(),
        inner_tokenizer: algorithm(Sentence, Seq[Word]),
        inner_stemmer: algorithm(Word, Stem),
    ):
        self.stopwords_remove = stopwords_remove
        self.inner_tokenizer = inner_tokenizer
        self.inner_stemmer = inner_stemmer

        SklearnTransformer.__init__(self)
        _CountVectorizer.__init__(self, lowercase=lowercase, binary=binary)

    def build_tokenizer(self):
        def func(sentence):
            tokens = self.inner_tokenizer.run(sentence)
            return [self.inner_stemmer.run(token) for token in tokens]

        return func

    def run(self, input: Seq[Sentence]) -> MatrixContinuousSparse:
        return SklearnTransformer.run(self, input)


class _FeatureVectorizer(SklearnTransformer):
    def __init__(self, sparse):
        self.vectorizer = DictVectorizer(sparse=sparse)
        super().__init__()

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)

    def transform(self, X, y=None):
        return self.vectorizer.transform(X)

@nice_repr
class FeatureSparseVectorizer(_FeatureVectorizer):
    def __init__(self):
        super().__init__(sparse=True)

    def run(self, input: Seq[FeatureSet]) -> MatrixContinuousSparse:
        return super().run(input)

@nice_repr
class FeatureDenseVectorizer(_FeatureVectorizer):
    def __init__(self):
        super().__init__(sparse=False)

    def run(self, input: Seq[FeatureSet]) -> MatrixContinuousDense:
        return super().run(input)

@nice_repr
class CRFTagger(CRF, SklearnEstimator):
    def __init__(
        self, algorithm: CategoricalValue("lbfgs", "l2sgd", "ap", "pa", "arow")
    ) -> None:
        SklearnEstimator.__init__(self)
        super().__init__(algorithm=algorithm)

    def run(
        self, X: Seq[Seq[FeatureSet]], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class ClassifierTagger(SklearnEstimator):
    """
    A wrapper class that uses a classifier as a tagger. This class is designed to work with classifiers, 
    possibly from sklearn, and use them as taggers. It handles the necessary transformations to fit and predict on sequence data.

    Attributes:
        classifier (algorithm): The classifier algorithm to use. 
        This should be a function that takes a MatrixContinuous and a Supervised[VectorCategorical] as input and returns a VectorCategorical.
    """
    def __init__(
        self, 
        classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)
    ) -> None:
        SklearnEstimator.__init__(self)
        self.classifier = classifier
        
    def _concatenate(self, items):
        return [embedding for sublist in items for embedding in sublist]
        
    def fit(self, X, y):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]

        # Concatenate X and y into single lists
        X_concat = self._concatenate(X)
        y_concat = [label for sublist in y for label in sublist]
        
        # Fit the classifier
        self.classifier.fit(X_concat, y_concat)
        return y

    def predict(self, X):
        # Concatenate X into a single list of embeddings
        X_concat = self._concatenate(X)
        
        # Predict using the classifier
        y_pred_concat = self.classifier.predict(X_concat)
        
        # Deconcatenate y_pred to match the original shape of X
        y_pred = [np.array([y_pred_concat[y_pos + i] for i in range(len(sub_X) if isinstance(sub_X, list) else sub_X.shape[0])]) for y_pos, sub_X in enumerate(X)]
            
        # Check the shapes of the predictions agains the original elements
        for i in range(len(X)):
            sy = np.shape(y_pred[i])
            sx = np.shape(X[i])
            assert sy[0] == sx[0]

        return y_pred

    def run(
        self, X: Seq[MatrixContinuousDense], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class SparseClassifierTagger(ClassifierTagger):
    def __init__(
        self, 
        classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)
    ) -> None:
        SklearnEstimator.__init__(self)
        self.classifier = classifier
        
    def _concatenate(self, items):
        sparse_matrices = [embedding for sublist in items for embedding in sublist]
        return vstack(sparse_matrices)

    def run(
        self, X: Seq[MatrixContinuousSparse], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class ClassifierTransformerTagger(SklearnEstimator):
    """
    A wrapper class that uses a classifier as a tagger. This class is designed to work with classifiers, 
    possibly from sklearn, and use them as taggers. It handles the necessary transformations to fit and predict on sequence data.

    Attributes:
        classifier (algorithm): The classifier algorithm to use. 
        This should be a function that takes a MatrixContinuous and a Supervised[VectorCategorical] as input and returns a VectorCategorical.
    """
    def __init__(
        self, 
        transformer: algorithm(MatrixContinuousDense, MatrixContinuousDense),
        classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)
    ) -> None:
        SklearnEstimator.__init__(self)
        self.classifier = classifier
        self.transformer = transformer
        
    def _concatenate(self, items):
        return [embedding for sublist in items for embedding in sublist]
        
    def fit(self, X, y):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]

        # Concatenate X and y into single lists
        X_concat = self._concatenate(X)
        y_concat = [label for sublist in y for label in sublist]
        
        if (self.transformer):
            X_concat = self.transformer.fit_transform(X_concat)
            
        # Fit the classifier
        self.classifier.fit(X_concat, y_concat)
        return y

    def predict(self, X):
        # Concatenate X into a single list of embeddings
        X_concat = self._concatenate(X)

        if (self.transformer):
            X_concat = self.transformer.fit_transform(X_concat)

        # Predict using the classifier
        y_pred_concat = self.classifier.predict(X_concat)

        # Deconcatenate y_pred to match the original shape of X
        y_pred = [np.array([y_pred_concat[y_pos + i] for i in range(len(sub_X) if isinstance(sub_X, list) else sub_X.shape[0])]) for y_pos, sub_X in enumerate(X)]

        return y_pred

    def run(
        self, X: Seq[MatrixContinuousDense], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class SparseClassifierTransformerTagger(ClassifierTransformerTagger):
    def __init__(
        self, 
        transformer: algorithm(MatrixContinuousSparse, MatrixContinuousSparse),
        classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)
    ) -> None:
        SklearnEstimator.__init__(self)
        self.classifier = classifier
        self.transformer = transformer
        
    def _concatenate(self, items):
        sparse_matrices = [embedding for sublist in items for embedding in sublist]
        return vstack(sparse_matrices)
        
    def run(
        self, X: Seq[MatrixContinuousSparse], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)

@nice_repr
class AggregatedTransformer(SklearnTransformer):
    def __init__(
        self, 
        transformer: algorithm(MatrixContinuousDense, MatrixContinuousDense)
    ) -> None:
        SklearnTransformer.__init__(self)
        self.transformer = transformer
        
    def _concatenate(self, items):
        return [embedding for sublist in items for embedding in sublist]
    
    def fit_transform(self, X, y=None):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]
        
        # Concatenate X and y into single lists
        X_concat = self._concatenate(X)
        
        if hasattr(self, "partial_fit"):
            self.partial_fit(X_concat)
        else:
            self.transformer.fit(X_concat)
            
        # call transform for each element in X
        return self.transform(X)

    def transform(self, X, y=None):
        return [self.transformer.transform(sub_X) for sub_X in X]
        
    def run(self, X: Seq[MatrixContinuousDense]) -> Seq[MatrixContinuousDense]:
        return SklearnTransformer.run(self, X)

@nice_repr
class SparseAggregatedTransformer(AggregatedTransformer):
    def __init__(
        self, 
        transformer: algorithm(MatrixContinuousSparse, MatrixContinuousSparse)
    ) -> None:
        SklearnTransformer.__init__(self)
        self.transformer = transformer
    
    def _concatenate(self, items):
        sparse_matrices = [embedding for sublist in items for embedding in sublist]
        return vstack(sparse_matrices)
        
    def run(self, X: Seq[MatrixContinuousSparse]) -> Seq[MatrixContinuousSparse]:
        return SklearnTransformer.run(self, X)


from sklearn.preprocessing import LabelEncoder as _LabelEncoder

class LabelEncoder(_LabelEncoder, SklearnTransformer):
    def __init__(
        self,
    ):
        SklearnTransformer.__init__(self)
        _LabelEncoder.__init__(
            self
        )

    def run(self, input: VectorCategorical) -> VectorDiscrete:
        return SklearnTransformer.run(self, input)

class CategoryDiscreteEncoderClassifier(_LabelEncoder, SklearnTransformer):
    def __init__(
        self, 
        encoder: algorithm(VectorCategorical, VectorDiscrete), # type: ignore
        classifier: algorithm(*[Seq[Sentence], Supervised[VectorDiscrete], VectorDiscrete]) # type: ignore
    ):
        self.encoder = encoder
        self.classifier = classifier
        SklearnTransformer.__init__(self)
        
    def _train(self, X, y=None):
        self.encoder._mode = "train"
        encoded_y = self.encoder.run(y)
        
        self.id_to_classes = {}
        self.classes_to_id = {}
        cls_ids = self.encoder.transform(self.encoder.classes_)
        
        for i in range(len(cls_ids)):
            cls = self.encoder.classes_[i]
            cls_id = cls_ids[i]
            self.id_to_classes[cls_id] = cls
            self.classes_to_id[cls] = cls_id
        
        self.classifier._mode = "train"
        pred_y = self.classifier.run(X, encoded_y)
        return [self.id_to_classes[cls_id] for cls_id in pred_y]

    def _eval(self, X, y=None):
        self.encoder._mode = "eval"
        encoded_y = self.encoder.run(y)
        
        self.id_to_classes = {}
        self.classes_to_id = {}
        cls_ids = self.encoder.transform(self.encoder.classes_)
        
        for i in range(len(cls_ids)):
            cls = self.encoder.classes_[i]
            cls_id = cls_ids[i]
            self.id_to_classes[cls_id] = cls
            self.classes_to_id[cls] = cls_id
        
        self.classifier._mode = "eval"
        pred_y = self.classifier.run(X, encoded_y)
        return [self.id_to_classes[cls_id] for cls_id in pred_y]

    def run(self, X: Seq[Sentence], y: Supervised[VectorCategorical]) -> VectorCategorical:
        return SklearnTransformer.run(self, X, y)


# @nice_repr
# class AggregatedVectorizer(SklearnTransformer):
#     def __init__(
#         self, 
#         vectorizer: algorithm(Seq[Sentence], MatrixContinuousDense)
#     ) -> None:
#         SklearnTransformer.__init__(self)
#         self.vectorizer = vectorizer
        
#     def _concatenate(self, items):
#         return [embedding for sublist in items for embedding in sublist]
    
#     def fit_transform(self, X, y=None):
#         # Save the starting index of each sublist in X
#         self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
#         self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]
        
#         # Concatenate X and y into single lists
#         X_concat = self._concatenate(X)
        
#         if hasattr(self, "partial_fit"):
#             self.partial_fit(X_concat)
#         else:
#             self.vectorizer.fit(X_concat)
            
#         # call transform for each element in X
#         return self.transform(X)

#     def transform(self, X, y=None):
#         return [self.vectorizer.transform(sub_X) for sub_X in X]
        
#     def run(self, X: Seq[Seq[Sentence]]) -> Seq[MatrixContinuousDense]:
#         return SklearnTransformer.run(self, X)
    
@nice_repr
class SparseAggregatedVectorizer(SklearnTransformer):
    def __init__(
        self, 
        vectorizer: algorithm(Seq[Sentence], MatrixContinuousSparse)
    ) -> None:
        SklearnTransformer.__init__(self)
        self.vectorizer = vectorizer
        
    def _concatenate(self, items):
        return [embedding for sublist in items for embedding in sublist]
        
    def fit_transform(self, X, y=None):
        # Save the starting index of each sublist in X
        self.concat_positions = [0] + [len(sublist) if isinstance(sublist, list) else sublist.shape[0] for sublist in X]
        self.concat_positions = np.cumsum(self.concat_positions).tolist()[:-1]
        
        # Concatenate X and y into single lists
        X_concat = self._concatenate(X)
        
        if hasattr(self, "partial_fit"):
            self.partial_fit(X_concat)
        else:
            self.vectorizer.fit(X_concat)
            
        # call transform for each element in X
        return self.transform(X)

    def transform(self, X, y=None):
        return [self.vectorizer.transform(sub_X) for sub_X in X]
    
    def run(self, X: Seq[Seq[Sentence]]) -> Seq[MatrixContinuousSparse]:
        return SklearnTransformer.run(self, X)


__all__ = [
    "CountVectorizerTokenizeStem",
    "FeatureSparseVectorizer",
    "FeatureDenseVectorizer",
    "CRFTagger",
    "ClassifierTagger",
    "SparseClassifierTagger",
    "ClassifierTransformerTagger",
    "SparseClassifierTransformerTagger",
    "PolynomialFeatures",
    # "AggregatedVectorizer",
    "SparseAggregatedVectorizer",
    "AggregatedTransformer",
    "SparseAggregatedTransformer"
]
