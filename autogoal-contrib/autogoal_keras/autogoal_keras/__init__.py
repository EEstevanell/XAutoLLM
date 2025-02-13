# NOTE: Reducing all the dumb tensorflow logs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from tensorflow import keras

    # assert keras.__version__ == "2.3.1"
except:
    print("(!) Code in `autogoal_keras` requires `keras`.")
    print("(!) You can install it with `pip install autogoal[keras]`.")
    raise


from autogoal_keras._base import (
    KerasClassifier,
    KerasSequenceClassifier,
    KerasSequenceTagger,
    KerasImageClassifier,
    KerasImagePreprocessor,
)
from autogoal_keras._grammars import build_grammar
