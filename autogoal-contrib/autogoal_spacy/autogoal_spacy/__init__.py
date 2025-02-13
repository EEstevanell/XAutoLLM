try:
    import spacy
except:
    print("(!) Code in `autogoal_spacy` requires `spacy`.")
    print("(!) You can install it with `pip install autogoal[spacy]`.")
    raise ImportError()


from autogoal_spacy._base import SpacyNLP
