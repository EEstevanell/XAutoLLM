try:
    import nltk
except:
    print("(!) Code in `autogoal_nltk` requires `nltk`.")
    print("(!) You can install it with `pip install autogoal[nltk]`.")
    raise

from autogoal_nltk._generated import *
from autogoal_nltk._manual import *

import os
from pathlib import Path


CONTRIB_NAME = "nltk"
DATA_PATH = Path.home() / ".autogoal" / "contrib" / CONTRIB_NAME / "data"


if DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, str(DATA_PATH))


def download():
    os.makedirs(DATA_PATH, exist_ok=True)
    return nltk.download(
        info_or_id=[
            "wordnet",
            "sentiwordnet",
            "averaged_perceptron_tagger",
            "rslp",
            "stopwords",
        ],
        download_dir=DATA_PATH,
    )


def status():
    from autogoal_contrib import ContribStatus

    try:
        from nltk.corpus import wordnet
        from nltk.corpus import sentiwordnet
        from nltk.corpus import stopwords

        from nltk.stem import RSLPStemmer

        st = RSLPStemmer()

        from nltk.tag import PerceptronTagger

        tagger = PerceptronTagger()
    except LookupError:
        return ContribStatus.RequiresDownload

    return ContribStatus.Ready
