from autogoal.datasets import semeval_2023_task_8_1 as semeval
from autogoal.datasets import intentsemo
from autogoal.datasets.semeval_2023_task_8_1 import F1_beta_plain, precision_plain, recall_plain, macro_f1_plain, macro_f1
from autogoal_sklearn._generated import MultinomialNB, MinMaxScaler, Perceptron, CountVectorizer, TfidfTransformer, KNNImputer
from autogoal_sklearn._manual import ClassifierTransformerTagger, ClassifierTagger, AggregatedTransformer, CountVectorizerFixedVocabulary
from autogoal_transformers._bert import BertEmbedding, BertSequenceEmbedding
# from autogoal_transformers._generated import TEC_Moritzlaurer_DebertaV3BaseMnliFeverAnli
from autogoal_transformers._tc_generated import TOC_Dslim_BertBaseNer
from autogoal_transformers._manual import SeqPretrainedTokenClassifier
# from autogoal_contrib import TuppleDeagregator, SparseMatrixConcatenator, DenseMatriConcatenator, AggregatedMatrixClassifier
from autogoal.kb import Seq, Tup, Word, VectorCategorical, VectorContinuous,MatrixContinuousSparse, MatrixCategorical,AggregatedMatrixContinuousSparse, Supervised, Tensor, Categorical, Dense, Label, Pipeline, Sentence
from autogoal.datasets.meddocan import F1_beta, precision, recall
from autogoal.ml import AutoML, peak_ram_usage, evaluation_time
from autogoal.search import RichLogger, NSPESearch, JsonLogger, PESearch
from autogoal_telegram import TelegramLogger
from autogoal.utils import Gb, Min, initialize_cuda_multiprocessing
from sklearn.model_selection import train_test_split

from autogoal_contrib import find_classes

def test_pipeline(data_size = 1000, opt = False):
    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Actual)
    
    X_train = X[:data_size]
    y_train = y[:data_size]
    
    X_test = X[data_size:2*data_size]
    y_test = y[data_size:2*data_size]
    
    pipeline = Pipeline(algorithms=
        [
            ClassifierTagger(classifier=Perceptron(
                    l1_ratio=0.15,
                    fit_intercept=True,
                    tol=0.001,
                    shuffle=True,
                    eta0=1,
                    early_stopping=False,
                    validation_fraction=0.1,
                    n_iter_no_change=5
                )) if opt else ClassifierTransformerTagger(transformer=MinMaxScaler(clip=True), classifier=MultinomialNB(fit_prior=False)),
        ],
        input_types=[Seq[Seq[Word]], Supervised[Seq[VectorCategorical]]],
    )
    
    pipeline.send("send")
    pipeline.run(X_train, y_train)
    
    pipeline.send("eval")
    predicted = pipeline.run(X_test, y_test)
    
    print(f"results: F1_beta {F1_beta(y_test, predicted)}, Precision {precision(y_test, predicted)}, Recall {recall(y_test, predicted)}")
    
def test_semeval_token_classification():
    X, y, X_test, y_test = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)
    
    a = AutoML(
        input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output=Seq[Seq[Label]],
        search_algorithm=NSPESearch,
        registry=[AggregatedTransformer, KNNImputer, Perceptron, ClassifierTagger, BertSequenceEmbedding],#find_classes(include="TOC") + [SeqPretrainedTokenClassifier],#[BertEmbedding, ClassifierTagger, ClassifierTransformerTagger, Perceptron, MultinomialNB, MinMaxScaler, TOC_Dslim_BertBaseNer],#,#[BertEmbedding, ClassifierTagger, ClassifierTransformerTagger, Perceptron, MultinomialNB, MinMaxScaler, Arbert_RobertaBaseFinetunedNerKmeansTwitter],
        objectives=(macro_f1, evaluation_time),
        maximize=(True, False),
        evaluation_timeout=20*Min,
        pop_size=10,
        memory_limit=20*Gb,
        search_timeout=5*Min
    )
    
    amount = 20

    X_train = X[:amount]
    y_train = y[:amount]
    
    X_test = X_test[:amount]
    y_test = y_test[:amount]
    
    loggers = [RichLogger()]
    a.fit(X_train, y_train, logger=loggers)
    
    results = a.predict(X_test)
    print(f"F1: {macro_f1(y_test, results)}, precision: {precision(y_test, results)}, recall: {recall(y_test, results)}")

def test_semeval_sentence_classification():
    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.SentenceClassification, data_option=semeval.SemevalDatasetSelection.Original, classes_mapping=semeval.TargetClassesMapping.Original)

    a = AutoML(
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        registry=find_classes(exclude="TEC"),
        objectives=(macro_f1_plain, peak_ram_usage),
        maximize=(True, False),
        evaluation_timeout=3*Min,
        search_timeout=10*Min,
        memory_limit=20*Gb
    )
    
    amount = 100
    
    X_train = X[:amount]
    y_train = y[:amount]
    
    X_test = X[amount:2*amount]
    y_test = y[amount:2*amount]
    
    loggers = [
        RichLogger(),
        JsonLogger("test.json"),
        TelegramLogger(token="6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M", channel="570734906", name="test", objectives=["Macro F1", ("RAM", "KB")])]
    a.fit(X_train, y_train, logger=loggers)
    
    results = a.score(X_test, y_test)
    print(f"F1: {results}")
    
def test_intentsemo_sentence_classification():
    X, y, _, _ = intentsemo.load(mode=intentsemo.TaskType.SentenceClassification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_0, X_train_1 = zip(*X_train)
    X_train_0, X_train_1 = list(X_train_0), list(X_train_1)
    
    X_test_0, X_test_1 = zip(*X_test)
    X_test_0, X_test_1 = list(X_test_0), list(X_test_1)

    a = AutoML(
        input=(Seq[Sentence], Seq[VectorContinuous], Supervised[VectorCategorical]),
        output=VectorCategorical,
        search_algorithm=PESearch,
        registry=[TuppleDeagregator, CountVectorizer, SparseMatrixConcatenator, DenseMatriConcatenator, MultinomialNB, AggregatedMatrixClassifier],
        objectives=macro_f1_plain,
        evaluation_timeout=3*Min,
        search_timeout=10*Min,
        memory_limit=20*Gb
    )
    
    loggers = [
        RichLogger(),
        JsonLogger("test.json")]
        # TelegramLogger(token="6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M", channel="570734906", name="test", objectives=["Macro F1", ("RAM", "KB")])]
    a.fit((X_train_0, X_train_1), y_train, logger=loggers)
    
    results = a.score((X_test, X_test), y_test)
    print(f"F1: {results}")
 
    
def get_token_dataset_properties():
    def get_proportions(data):
        classes = ['claim', 'per_exp', 'claim_per_exp', 'question', 'O']
        proportions = [0] * len(classes)
        for sample in data:
            for i, cls in enumerate(classes):
                proportions[i] += sample.count(cls)
        total_samples = sum([len(di) for di in data])
        average_proportions = [count / total_samples for count in proportions]
        return average_proportions

    X, y, X_test, y_test = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)
    
    # print(len(X), len(X_test))
    
    # get average len of X + X_test
    # print(sum([len(x) for x in X + X_test])/(len(X) + len(X_test)))
    # print(sum([len(x) for x in X])/(len(X)) )
    # print(sum([len(x) for x in X_test])/(len(X_test)))
    
    # get average proportions of each class in X and X_test 
    
    y_average_proportions = get_proportions(y)
    y_test_average_proportions = get_proportions(y_test)
    print("Average proportions in y:", y_average_proportions)
    print("Average proportions in y_test:", y_test_average_proportions)
    
    import plotly.graph_objects as go
    import plotly.io as py
    # Assuming your data is structured like this:
    x = ['claim', 'per_exp', 'claim_per_exp', 'question', 'O']
    # Create a trace for each dataset
    trace1 = go.Bar(x=x, y=y_average_proportions, name='Train Split')
    trace2 = go.Bar(x=x, y=y_test_average_proportions, name='Test Split')
    # Create the figure and add the traces
    fig = go.Figure(data=[trace1, trace2])
    # Update the layout to have the barmode be 'group's
    fig.update_layout(
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.2, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    # Save the figure as a jpg image
    py.write_image(fig, 'token_data_proportions.jpg', scale=3)
            
def get_sentence_dataset_properties():
    def get_proportions(data):
        classes = ['claim', 'per_exp', 'claim_per_exp', 'question', 'O']
        proportions = [0] * len(classes)
        for i, cls in enumerate(classes):
            proportions[i] += data.count(cls)
        total_samples = len(data)
        average_proportions = [count / total_samples for count in proportions]
        return average_proportions

    X, y, _, _ = semeval.load(mode=semeval.TaskTypeSemeval.SentenceClassification, data_option=semeval.SemevalDatasetSelection.Original, classes_mapping=semeval.TargetClassesMapping.Original)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(len(X), len(X_test))
    
    # get average len of X + X_test
    print(sum([len(x) for x in X + X_test])/(len(X) + len(X_test)))
    print(sum([len(x) for x in X])/(len(X)) )
    print(sum([len(x) for x in X_test])/(len(X_test)))
    
    # get average proportions of each class in X and X_test 
    
    y_average_proportions = get_proportions(y)
    y_test_average_proportions = get_proportions(y_test)
    print("Average proportions in y:", y_average_proportions)
    print("Average proportions in y_test:", y_test_average_proportions)
    
    import plotly.graph_objects as go
    import plotly.io as py
    # Assuming your data is structured like this:
    x = ['claim', 'per_exp', 'claim_per_exp', 'question', 'O']
    # Create a trace for each dataset
    trace1 = go.Bar(x=x, y=y_average_proportions, name='Train Split')
    trace2 = go.Bar(x=x, y=y_test_average_proportions, name='Test Split')
    # Create the figure and add the traces
    fig = go.Figure(data=[trace1, trace2])
    # Update the layout to have the barmode be 'group's
    fig.update_layout(
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.2, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    # Save the figure as a jpg image
    py.write_image(fig, 'sentence_data_proportions.jpg', scale=3)
    
    
    

if __name__ == '__main__':
    # get_sentence_dataset_properties()
    # get_token_dataset_properties()
    
    # BertTokenizeEmbedding.download()
    # BertEmbedding.download()
    
    # test_pipeline(3000, True)
    # test_pipeline(3000)
    # from autogoal.utils._process import initialize_cuda_multiprocessing
    
    # initialize_cuda_multiprocessing()
    # test_semeval_token_classification()
    # test_semeval_sentence_classification()
    test_intentsemo_sentence_classification()
    
    # def get_proportions(list):
    #     # Calculate the sum of each component across all tuples
    #     component_sums = [sum(tup[i] for tup in list) for i in range(len(list[0]))]
    #     # Calculate the total sum of all components
    #     total_sum = sum(component_sums)
    #     # Calculate the proportion of each component

    # import numpy as np
    # from deap import base, creator, tools, algorithms
    # import random
    # import sys
    
    # # print(len(y))
    # # print(len(selected_y))
    
    # # X_train, y_train, X_test, y_test = semeval.load(mode=semeval.TaskTypeSemeval.TokenClassification, data_option=semeval.SemevalDatasetSelection.Original)
    # # print(len(X_train)/len(X_test))
    
    
    # classes = ['O', 'claim', 'per_exp', 'claim_per_exp', 'question']
    
    # original_list = []
    # for yi in y:
    #     counts = tuple(yi.count(cls) for cls in classes)
    #     counts = tuple(count / sum(counts) for count in counts)
    #     original_list.append(counts)
        
    #     try:
    #         return [comp_sum / total_sum for comp_sum in component_sums]
    #     except ZeroDivisionError:
    #         return [0 for comp_sum in component_sums]
    
    # def euclidean_distance(a, b):
    #     return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

    # def func(original_list, selected_indexes):
    #     if len(selected_indexes) == 0:
    #         return sys.float_info.max
    
    #     notselected = [original_list[i] for i in range(len(original_list)) if i not in selected_indexes]
    #     selected = [original_list[i] for i in selected_indexes]
        
    #     notselected_proportions = get_proportions(notselected)
    #     selected_proportions = get_proportions(selected)
        
    #     return euclidean_distance(notselected_proportions, selected_proportions)
        
        
    # # Define the fitness function
    # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMin)

    # toolbox = base.Toolbox()
    
    # # Attribute generator 
    # toolbox.register("index", random.randint, 0, len(original_list)-1)

    # # Structure initializers
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.index, n=int(len(original_list)*0.2))
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # def evalFunc(individual):
    #     return func(original_list, individual),

    # # Operator registering
    # toolbox.register("evaluate", evalFunc)
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(original_list)-1, indpb=0.05)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # def main():
    #     pop = toolbox.population(n=50)
    #     hof = tools.HallOfFame(1)
    #     stats = tools.Statistics(lambda ind: ind.fitness.values)
    #     stats.register("avg", np.mean)
    #     stats.register("min", np.min)
    #     stats.register("max", np.max)
        
    #     pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)\
        
    #     best_indexes = hof[0]
    #     best_list = [original_list[i] for i in best_indexes]
        
    #     with open("best_indexes.txt", "w") as f:
    #         import json
    #         json.dump(best_indexes, f)
        
    #     return pop, logbook, hof
    
    # print(main())
