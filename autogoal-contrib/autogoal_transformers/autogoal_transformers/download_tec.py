from autogoal_contrib import find_classes
from autogoal_transformers import BertEmbedding

BertEmbedding.download()

for alg in find_classes("TEC"):
    try:
        alg.download()
    except:
        print(f"Failed to download {alg.__name__}")