from autogoal.datasets import imdb_50k_movie_reviews as movie_reviews
from autogoal.utils import Min, Gb
from autogoal.ml import AutoML, evaluation_time
from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
from autogoal.kb import Seq, Supervised, VectorCategorical, Sentence
from autogoal_sklearn._generated import (
    MultinomialNB,
    PassiveAggressiveClassifier,
    MinMaxScaler,
    Perceptron,
    KNNImputer,
)
from autogoal_contrib import find_classes
from autogoal.search import RichLogger
from autogoal.search import NSPESearch


def test_pipeline():
    X_train, X_test, y_train, y_test = movie_reviews.load()

    model = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=find_classes(include="TEXT_GEN|SEQ_EMB")
        + [
            MultinomialNB,
            PassiveAggressiveClassifier,
            MinMaxScaler,
            Perceptron,
            KNNImputer,
        ],
        objectives=(macro_f1_plain, evaluation_time),
        search_algorithm=NSPESearch,
        maximize=(True, False),
        evaluation_timeout=5 * Min,
        search_timeout=10 * Min,
        memory_limit=20 * Gb,
        cross_validation_steps=1,
    )

    loggers = [
        RichLogger(),
        # TelegramLogger(
        #     token="6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
        #     channel="570734906",
        #     name="test",
        #     objectives=["Macro F1", ("Eval Time", "Seconds")],
        # ),
    ]

    model.fit(X_train, y_train, logger=loggers)
    results = model.score(X_test, y_test)

    print(f"F1: {results}")


if __name__ == "__main__":
    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    test_pipeline()
