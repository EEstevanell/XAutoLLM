from typing import Tuple
from autogoal.kb import *
from autogoal.grammar import CategoricalValue, BooleanValue
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase
from scipy import sparse
import numpy as np

@nice_repr
class VectorAggregator(AlgorithmBase):
    def __init__(self, mode: CategoricalValue("mean", "max")):
        self.mode = mode

    def run(self, input: Seq[VectorContinuous]) -> VectorContinuous:
        input = np.vstack(input)

        if self.mode == "mean":
            return input.mean(axis=1)
        elif self.mode == "max":
            return input.max(axis=1)

        raise ValueError("Invalid mode: %s" % self.mode)

@nice_repr
class DenseMatriConcatenator(AlgorithmBase):
    def __init__(self):
        pass

    def run(self, input: MatrixContinuousDense, input2: Seq[VectorContinuous]) -> MatrixContinuousDense:
        # Convert the sequence of vectors to a dense matrix
        input2_matrix = np.array(input2)
        
        # Concatenate the two matrices
        result = np.concatenate([input, input2_matrix], axis=1)
        return result
    
@nice_repr
class DenseClassifier(AlgorithmBase):
    def __init__(self, classifier: algorithm(MatrixContinuous, Supervised[VectorCategorical], VectorCategorical)):
        super().__init__()
        self.classifier = classifier
        
    def train(self):
        return self.classifier.train()
    
    def eval(self):
        return self.classifier.eval()
        
    def run(self, X: MatrixContinuousDense, y: Supervised[VectorCategorical]) -> VectorCategorical:
        return self.classifier.run(X, y)

@nice_repr
class MatrixBuilder(AlgorithmBase):
    """
    Builds a matrix from a list of vectors.

    ##### Examples

    ```python
    >>> import numpy as np
    >>> x1 = np.asarray([1,2,3])
    >>> x2 = np.asarray([2,3,4])
    >>> x3 = np.asarray([3,4,5])
    >>> MatrixBuilder().run([x1, x2, x3])
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    ```
    """

    def run(self, input: Seq[VectorContinuous]) -> MatrixContinuousDense:
        return np.vstack(input)


@nice_repr
class TensorBuilder(AlgorithmBase):
    """
    Builds a 3D tensor from a list of matrices.

    ##### Examples

    ```python
    >>> import numpy as np
    >>> x1 = np.asarray([[1,2],[3,4]])
    >>> x2 = np.asarray([[2,3],[4,5]])
    >>> x3 = np.asarray([[3,4],[5,6]])
    >>> TensorBuilder().run([x1, x2, x3])
    array([[[1, 2],
            [3, 4]],
    <BLANKLINE>
           [[2, 3],
            [4, 5]],
    <BLANKLINE>
           [[3, 4],
            [5, 6]]])

    ```
    """

    def run(self, input: Seq[MatrixContinuousDense]) -> Tensor3:
        return np.vstack([np.expand_dims(m, axis=0) for m in input])


@nice_repr
class FlagsMerger(AlgorithmBase):
    def run(self, input: Seq[FeatureSet]) -> FeatureSet:
        result = {}

        for d in input:
            result.update(d)

        return result


@nice_repr
class MultipleFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        extractors: Distinct(
            algorithm(Word, FeatureSet), exceptions=["MultipleFeatureExtractor", "SentenceFeatureExtractor", "DocumentFeatureExtractor"]
        ),
        merger: algorithm(Seq[FeatureSet], FeatureSet),
    ):
        self.extractors = [extractor() for extractor in extractors]
        self.merger = merger

    def run(self, input: Word) -> FeatureSet:
        flags = []
        
        for extractor in self.extractors:
            flags.append(extractor.run(input))
            
        return self.merger.run(flags)

@nice_repr
class SentenceFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        tokenizer: algorithm(Sentence, Seq[Word]),
        feature_extractor: algorithm(Word, FeatureSet, exceptions=["SentenceFeatureExtractor"]),
        include_text: BooleanValue(),
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.include_text = include_text

    def run(self, input: Sentence) -> FeatureSet:
        tokens = self.tokenizer.run(input)
        
        flags = []
        for w in tokens:
            flags.append(self.feature_extractor.run(w))

        if self.include_text:
            return {
                f"{w}|{f}": v for w, flag in zip(tokens, flags) for f, v in flag.items()
            }
        else:
            return {f: v for flag in flags for f, v in flag.items()}

@nice_repr
class SentenceSeqWordFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        tokenizer: algorithm(Sentence, Seq[Word]),
        feature_extractor: algorithm(Seq[Word], Seq[FeatureSet]),
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def run(self, input: Sentence) -> Seq[FeatureSet]:
        tokens = self.tokenizer.run(input)
        return self.feature_extractor.run(tokens)

@nice_repr
class DocumentFeatureExtractor(AlgorithmBase):
    def __init__(
        self,
        tokenizer: algorithm(Document, Seq[Sentence]),
        feature_extractor: algorithm(Sentence, FeatureSet, exceptions=["DocumentFeatureExtractor"]),
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def run(self, input: Document) -> Seq[FeatureSet]:
        tokens = self.tokenizer.run(input)
        flags = [self.feature_extractor.run(w) for w in tokens]
        return flags

@nice_repr
class DocumentAsSentence(AlgorithmBase):
    def run(self, input: Document) -> Sentence:
        return input