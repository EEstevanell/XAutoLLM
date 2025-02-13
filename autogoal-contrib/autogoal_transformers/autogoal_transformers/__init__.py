import os
from pathlib import Path

from inspect import getsourcefile
from os.path import abspath, dirname, join

# DATA_PATH should be the main directory of autogoal_transformers contrib
DATA_PATH = dirname(abspath(getsourcefile(lambda: 0)))

# Setting up download location
os.environ["TRANSFORMERS_CACHE"] = str(DATA_PATH)
os.environ["HF_HOME"] = str(DATA_PATH)


try:
    import torch
    import transformers
except:
    print("(!) Code in `autogoal_transformers` requires `pytorch` and `transformers`.")
    print("(!) You can install it with `pip install autogoal[transformers]`.")
    raise


from autogoal_transformers._bert import BertEmbedding, BertSequenceEmbedding, BertTokenizeSequenceEmbedding
from autogoal_transformers._generated import *
from autogoal_transformers._manual import *

def download():
    BertEmbedding.download()
    return True


def status():
    from autogoal_contrib import ContribStatus

    try:
        BertEmbedding.check_files()
    except OSError:
        return ContribStatus.RequiresDownload

    return ContribStatus.Ready
