import math
from multiprocessing import Process
from autogoal.datasets import sst2, ag_news, meld, liar
from autogoal.meta_learning._logging import ExperienceLogger
from autogoal.meta_learning.distance import (
    CosineDistance,
    EuclideanDistance,
)
from autogoal.meta_learning.normalization import (
    StandardScalerNormalizer,
)
from autogoal.meta_learning.warm_start import WarmStart
from autogoal.search._warm_start_pge import NSPEWarmStartSearch
from autogoal.utils import Gb, Hour

random_seed = 42


def set_cpu_affinity(cpu_list):
    import psutil

    p = psutil.Process()
    p.cpu_affinity(cpu_list)

def evenly_split_cpus(num_experiments, total_cpus=None):
    import os

    if total_cpus is None:
        total_cpus = os.cpu_count()

    if num_experiments == 1:
        return [list(range(total_cpus))]

    cpus_per_experiment = total_cpus // num_experiments
    extra_cpus = total_cpus % num_experiments

    cpu_lists = []
    start_cpu = 0
    for i in range(num_experiments):
        # Distribute the extra CPUs one by one to the first few experiments
        cpus_for_this_experiment = cpus_per_experiment + (1 if i < extra_cpus else 0)
        end_cpu = start_cpu + cpus_for_this_experiment
        cpu_list = list(range(start_cpu, end_cpu))
        cpu_lists.append(cpu_list)
        start_cpu = end_cpu

    return cpu_lists

def warmstart_search(
    id,
    dataset,
    dataset_name,
    device_id,
    cpu_list=None,
    random_seed=42,
    warms_start_args=dict(),
    bot_token="6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
):
    # Import necessary modules within the function
    import os
    import sys
    import torch
    import numpy as np
    import random
    from autogoal.meta_learning.warm_start import WarmStart
    from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
    from autogoal.ml import AutoML, evaluation_time, accuracy
    from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
    from autogoal.kb import Seq, Supervised, VectorDiscrete, Sentence
    from autogoal_contrib import find_classes
    from autogoal.search import JsonLogger, ConsoleLogger
    from autogoal.search import NSPESearch
    from autogoal_telegram import TelegramLogger
    from autogoal_transformers._manual import (
        FineTuneGenLLMClassifier,
    )
    from autogoal_transformers._manual import (
        FineTuneGenLLMClassifier,
        FineTuneLLMEmbeddingClassifier,
        LoraGenLLMClassifier,
        LoraLLMEmbeddingClassifier,
        PartialFineTuneGenLLMClassifier,
        PartialFineTuneLLMEmbeddingClassifier,
    )

    if cpu_list is not None:
        set_cpu_affinity(cpu_list)

    # Set the CUDA device at the very beginning
    torch.cuda.set_device(device_id)

    # Redirect stdout and stderr to separate log files
    log_filename = f"{dataset_name}_id_{id}.out"
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    # Verification code
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Process for {dataset_name} is using device {current_device}: {device_name}")
    print(f"Process for {dataset_name} is using CPUs: {cpu_list}")

    # Set environment variables for threading libraries
    cpu_count = len(cpu_list)
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["BLIS_NUM_THREADS"] = str(cpu_count)

    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Load your dataset
    X_train, y_train, X_test, y_test = dataset.load()

    # Initialize the WarmStart object
    warm_start = WarmStart(
        positive_min_threshold=warms_start_args.get("positive_min_threshold", 0.2),
        adaptative_negative_alpha_limit=warms_start_args.get(
            "adaptative_negative_alpha_limit", None
        ),
        k_pos=warms_start_args.get("k_pos", 10),  # Number of experiences to consider
        k_neg=warms_start_args.get("k_neg", 10),  # Number of experiences to consider
        distance=warms_start_args.get("distance", EuclideanDistance),
        normalizers=warms_start_args.get(
            "normalizers", [LogNormalizer(), MinMaxNormalizer()]
        ),
        exclude=f"{warms_start_args.get('exclude', '')}|_warmstart",
        include=warms_start_args.get("include", None),
        beta_scale=warms_start_args.get("beta_scale", 1.0),
        beta=warms_start_args.get("beta", None),
        max_alpha=warms_start_args.get("max_alpha", 0.05),
        min_alpha=warms_start_args.get("min_alpha", -0.02),
    )
    warm_start.pre_warm_up(X_train, y_train)

    model = AutoML(
        input=(Seq[Sentence], Supervised[VectorDiscrete]),
        output=VectorDiscrete,
        random_state=random_seed,
        registry=[
            FineTuneLLMEmbeddingClassifier,
            PartialFineTuneLLMEmbeddingClassifier,
            LoraLLMEmbeddingClassifier,
        ]
        + find_classes(include="WORD_EMB")
        + [
            FineTuneGenLLMClassifier,
            PartialFineTuneGenLLMClassifier,
            LoraGenLLMClassifier,
        ]
        + find_classes(include="TEXT_GEN"),
        evaluation_timeout=1.5 * Hour,
        search_timeout=24 * Hour,
        memory_limit=35 * Gb,
        cross_validation_steps=2,
        stratified_cross_validation=True,
        objectives=(macro_f1_plain, evaluation_time),
        observations=[("Accuracy", accuracy)],
        maximize=(True, False),
        search_algorithm=NSPEWarmStartSearch,
        warm_start=warm_start,
    )

    # Initialize loggers
    loggers = [
        ConsoleLogger(),
        JsonLogger(f"titan-{dataset_name}-warm-start-id:{id}.json"),
        ExperienceLogger(
            dataset_features=warm_start.current_dataset_features,
            system_features=warm_start.current_system_features,
            dataset_feature_extractor_name="TextClassificationFeatureExtractor",
            system_feature_extractor_name="SystemFeatureExtractor",
            alias=f"{dataset_name}_warmstart_id:{id} (rn:{random_seed})",
        ),
        TelegramLogger(
            token=bot_token,
            channel="570734906",
            name=f"Titan {dataset_name} - Warmstart - Id:{id}",
            objectives=["Macro F1", ("Eval Time", "Seconds")],
        ),
    ]

    # Fit the model and evaluate
    model.fit(X_train, y_train, logger=loggers)
    results = model.score(X_test, y_test)

    print(f"{dataset_name} F1: {results}")

def warmstart_search_single_objective(
    id,
    dataset,
    dataset_name,
    device_id,
    cpu_list=None,
    random_seed=42,
    warms_start_args=None,
    bot_token="6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
):
    # Import necessary modules within the function
    import os
    import sys
    import torch
    import numpy as np
    import random
    from autogoal.meta_learning.warm_start import WarmStart
    from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
    from autogoal.ml import AutoML, evaluation_time, accuracy
    from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
    from autogoal.kb import Seq, Supervised, VectorDiscrete, Sentence
    from autogoal_contrib import find_classes
    from autogoal.search import JsonLogger, ConsoleLogger
    from autogoal.search import NSPESearch
    from autogoal_telegram import TelegramLogger
    from autogoal_transformers._manual import (
        FineTuneGenLLMClassifier,
    )
    from autogoal_transformers._manual import (
        FineTuneGenLLMClassifier,
        FineTuneLLMEmbeddingClassifier,
        LoraGenLLMClassifier,
        LoraLLMEmbeddingClassifier,
        PartialFineTuneGenLLMClassifier,
        PartialFineTuneLLMEmbeddingClassifier,
    )

    if cpu_list is not None:
        set_cpu_affinity(cpu_list)

    # Set the CUDA device at the very beginning
    torch.cuda.set_device(device_id)

    # Redirect stdout and stderr to separate log files
    log_filename = f"{dataset_name}_id_{id}.out"
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    # Verification code
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Process for {dataset_name} is using device {current_device}: {device_name}")
    print(f"Process for {dataset_name} is using CPUs: {cpu_list}")

    # Set environment variables for threading libraries
    cpu_count = len(cpu_list)
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["BLIS_NUM_THREADS"] = str(cpu_count)

    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Using Random Seed: {seed}")

    # Load your dataset
    X_train, y_train, X_test, y_test = dataset.load()

    # Initialize the WarmStart object
    warm_start = (
        WarmStart(
            positive_min_threshold=warms_start_args.get("positive_min_threshold", 0.2),
            adaptative_negative_alpha_limit=warms_start_args.get(
                "adaptative_negative_alpha_limit", None
            ),
            adaptative_positive_alpha_limit=warms_start_args.get(
                "adaptative_positive_alpha_limit", None
            ),
            k_pos=warms_start_args.get(
                "k_pos", 10
            ),  # Number of experiences to consider
            k_neg=warms_start_args.get(
                "k_neg", 10
            ),  # Number of experiences to consider
            distance=warms_start_args.get("distance", EuclideanDistance),
            normalizers=warms_start_args.get(
                "normalizers", [LogNormalizer(), MinMaxNormalizer()]
            ),
            exclude=f"{warms_start_args.get('exclude', '')}|_warmstart",
            include=warms_start_args.get("include", None),
            beta_scale=warms_start_args.get("beta_scale", 1.0),
            beta=warms_start_args.get("beta", None),
            max_alpha=warms_start_args.get("max_alpha", 0.05),
            min_alpha=warms_start_args.get("min_alpha", -0.02),
            f1_weight=warms_start_args.get("f1_weight", 0.5),
            evaluation_time_weight=warms_start_args.get("evaluation_time_weight", 0.5),
            utility_function=warms_start_args.get("utility_function", None),
        )
        if warms_start_args is not None
        else WarmStart()
    )

    warm_start.pre_warm_up(X_train, y_train)

    model = (
        AutoML(
            input=(Seq[Sentence], Supervised[VectorDiscrete]),
            output=VectorDiscrete,
            random_state=random_seed,
            registry=[
                FineTuneLLMEmbeddingClassifier,
                PartialFineTuneLLMEmbeddingClassifier,
                LoraLLMEmbeddingClassifier,
            ]
            + find_classes(include="WORD_EMB")
            + [
                FineTuneGenLLMClassifier,
                PartialFineTuneGenLLMClassifier,
                LoraGenLLMClassifier,
            ]
            + find_classes(include="TEXT_GEN"),
            evaluation_timeout=1.5 * Hour,
            search_timeout=24 * Hour,
            memory_limit=35 * Gb,
            cross_validation_steps=2,
            stratified_cross_validation=True,
            objectives=(macro_f1_plain,),
            observations=[("Accuracy", accuracy)],
            maximize=(True,),
            search_algorithm=NSPEWarmStartSearch,
            warm_start=warm_start,
        )
        if warms_start_args is not None
        else AutoML(
            input=(Seq[Sentence], Supervised[VectorDiscrete]),
            output=VectorDiscrete,
            random_state=random_seed,
            registry=[
                FineTuneLLMEmbeddingClassifier,
                PartialFineTuneLLMEmbeddingClassifier,
                LoraLLMEmbeddingClassifier,
            ]
            + find_classes(include="WORD_EMB")
            + [
                FineTuneGenLLMClassifier,
                PartialFineTuneGenLLMClassifier,
                LoraGenLLMClassifier,
            ]
            + find_classes(include="TEXT_GEN"),
            evaluation_timeout=1.5 * Hour,
            search_timeout=24 * Hour,
            memory_limit=35 * Gb,
            cross_validation_steps=2,
            stratified_cross_validation=True,
            objectives=(macro_f1_plain,),
            observations=[("Accuracy", accuracy)],
            maximize=(True,),
            search_algorithm=NSPESearch,
        )
    )

    # Initialize loggers
    loggers = [
        ConsoleLogger(),
        JsonLogger(f"titan-{dataset_name}-warm-start-id:{id}-single-objective.json"),
        ExperienceLogger(
            dataset_features=warm_start.current_dataset_features,
            system_features=warm_start.current_system_features,
            dataset_feature_extractor_name="TextClassificationFeatureExtractor",
            system_feature_extractor_name="SystemFeatureExtractor",
            alias=(
                f"{dataset_name}_warmstart_id:{id} (rn:{random_seed})"
                if warms_start_args is not None
                else f"{dataset_name}_id:{id} (rn:{random_seed})"
            ),
        ),
        TelegramLogger(
            token=bot_token,
            channel="570734906",
            name=f"Titan {dataset_name} - Warmstart - Id:{id} - single-objective",
            objectives=["Macro F1"],
        ),
    ]

    # Fit the model and evaluate
    set_seed(random_seed)
    model.fit(X_train, y_train, logger=loggers)
    results = model.score(X_test, y_test)

    print(f"{dataset_name} F1: {results}")

def regular(dataset, dataset_name, device_id, cpu_list=None, random_seed=42):
    # Import necessary modules within the function
    import os
    import sys
    import torch
    import numpy as np
    import random
    from autogoal.meta_learning.warm_start import WarmStart
    from autogoal.meta_learning.normalization import LogNormalizer, MinMaxNormalizer
    from autogoal.ml import AutoML, evaluation_time, accuracy
    from autogoal.datasets.semeval_2023_task_8_1 import macro_f1_plain
    from autogoal.kb import Seq, Supervised, VectorDiscrete, Sentence
    from autogoal_contrib import find_classes
    from autogoal.search import JsonLogger, ConsoleLogger
    from autogoal.search import NSPESearch
    from autogoal_telegram import TelegramLogger
    from autogoal_transformers._manual import (
        FineTuneGenLLMClassifier,
    )
    from autogoal_transformers._manual import (
        FineTuneGenLLMClassifier,
        FineTuneLLMEmbeddingClassifier,
        LoraGenLLMClassifier,
        LoraLLMEmbeddingClassifier,
        PartialFineTuneGenLLMClassifier,
        PartialFineTuneLLMEmbeddingClassifier,
    )

    if cpu_list is not None:
        set_cpu_affinity(cpu_list)

    # Set the CUDA device at the very beginning
    torch.cuda.set_device(device_id)

    # Redirect stdout and stderr to separate log files
    log_filename = f"{dataset_name}_device_{device_id}.out"
    sys.stdout = open(log_filename, "w")
    sys.stderr = sys.stdout

    # Verification code
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Process for {dataset_name} is using device {current_device}: {device_name}")
    print(f"Process for {dataset_name} is using CPUs: {cpu_list}")

    # Set environment variables for threading libraries
    cpu_count = len(cpu_list)
    os.environ["OMP_NUM_THREADS"] = str(cpu_count)
    os.environ["MKL_NUM_THREADS"] = str(cpu_count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
    os.environ["BLIS_NUM_THREADS"] = str(cpu_count)

    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Load your dataset
    X_train, y_train, X_test, y_test = dataset.load()

    # Initialize the WarmStart object
    warm_start = WarmStart(
        positive_min_threshold=0.2,
        k_pos=10,  # Number of experiences to consider
        k_neg=10,  # Number of experiences to consider
        normalizers=[LogNormalizer(), MinMaxNormalizer()],
    )
    warm_start.pre_warm_up(X_train, y_train)

    print(y_train[0])
    model = AutoML(
        input=(Seq[Sentence], Supervised[VectorDiscrete]),
        output=VectorDiscrete,
        random_state=random_seed,
        registry=[
            FineTuneLLMEmbeddingClassifier,
            PartialFineTuneLLMEmbeddingClassifier,
            LoraLLMEmbeddingClassifier,
        ]
        + find_classes(include="WORD_EMB")
        + [
            FineTuneGenLLMClassifier,
            PartialFineTuneGenLLMClassifier,
            LoraGenLLMClassifier,
        ]
        + find_classes(include="TEXT_GEN"),
        evaluation_timeout=1.5 * Hour,
        search_timeout=48 * Hour,
        memory_limit=35 * Gb,
        cross_validation_steps=2,
        stratified_cross_validation=True,
        objectives=(macro_f1_plain, evaluation_time),
        observations=[("Accuracy", accuracy)],
        maximize=(True, False),
        search_algorithm=NSPESearch,
    )

    # Initialize loggers
    loggers = [
        ConsoleLogger(),
        JsonLogger(f"titan-{dataset_name}-warm-start.json"),
        ExperienceLogger(
            dataset_features=warm_start.current_dataset_features,
            system_features=warm_start.current_system_features,
            dataset_feature_extractor_name="TextClassificationFeatureExtractor",
            system_feature_extractor_name="SystemFeatureExtractor",
            alias=dataset_name,
        ),
        TelegramLogger(
            token="6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
            channel="570734906",
            name=f"Titan {dataset_name} - Regular",
            objectives=["Macro F1", ("Eval Time", "Seconds")],
        ),
    ]

    # Fit the model and evaluate
    model.fit(X_train, y_train, logger=loggers)
    results = model.score(X_test, y_test)

    print(f"{dataset_name} F1: {results}")

def run_baselines():
    # Define the datasets and corresponding device IDs
    grouped_experiments = [
        [
            # round 1
            (ag_news, "ag_news", 0),  # Use CUDA device 0
            (sst2, "sst2", 1),  # Use CUDA device 1
        ],
        [
            # round 2
            (liar, "liar", 0),  # Use CUDA device 0
            (meld, "meld", 1),  # Use CUDA device 1
        ],
    ]

    for experiments in grouped_experiments:
        num_experiments = len(experiments)

        # Get CPU core lists for each experiment
        cpu_lists = evenly_split_cpus(num_experiments)

        processes = []
        for idx, (dataset, name, device_id) in enumerate(experiments):
            cpu_list = cpu_lists[idx]
            p = Process(target=regular, args=(dataset, name, device_id, cpu_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

def run_liar_metalearning_test():
    # Initialize experiments list
    grouped_experiment_pairs = [
        [
            (
                (
                    3,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    0,  # CUDA device 0
                    "6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": EuclideanDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0.5,
                        "exclude": "liar|warmstart",
                    },
                ),
                (
                    4,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    1,  # CUDA device 1
                    "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0.5,
                        "exclude": "liar|warmstart",
                    },
                ),
            ),
            (
                (
                    5,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    0,  # CUDA device 0
                    "6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": EuclideanDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0,
                        "exclude": "liar|warmstart",
                    },
                ),
                (
                    6,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    1,  # CUDA device 1
                    "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0,
                        "exclude": "liar|warmstart",
                    },
                ),
            ),
            (
                (
                    7,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    0,  # CUDA device 0
                    "6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": EuclideanDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 1,
                        "exclude": "liar|warmstart",
                    },
                ),
                (
                    8,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    1,  # CUDA device 1
                    "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 1,
                        "exclude": "liar|warmstart",
                    },
                ),
            ),
        ],
        [
            (
                (
                    5,
                    sst2,  # Dataset object assumed to be defined elsewhere
                    "sst2",
                    0,  # CUDA device 0
                    "6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": EuclideanDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0,
                        "exclude": "sst2|warmstart",
                    },
                ),
                (
                    6,
                    sst2,  # Dataset object assumed to be defined elsewhere
                    "sst2",
                    1,  # CUDA device 1
                    "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0,
                        "exclude": "sst2|warmstart",
                    },
                ),
            ),
            (
                (
                    7,
                    sst2,  # Dataset object assumed to be defined elsewhere
                    "sst2",
                    0,  # CUDA device 0
                    "6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": EuclideanDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 1,
                        "exclude": "sst2|warmstart",
                    },
                ),
                (
                    8,
                    sst2,  # Dataset object assumed to be defined elsewhere
                    "sst2",
                    1,  # CUDA device 1
                    "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scaler": 1,
                        "exclude": "sst2|warmstart",
                    },
                ),
            ),
        ],
    ]

    grouped_experiment_pairs_k_0 = [
        (
            (
                (
                    6,
                    liar,  # Dataset object assumed to be defined elsewhere
                    "liar",
                    0,  # CUDA device 0
                    "6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0,
                        "exclude": "liar|warmstart",
                    },
                ),
                (
                    6,
                    sst2,  # Dataset object assumed to be defined elsewhere
                    "sst2",
                    1,  # CUDA device 1
                    "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M",
                    {  # WarmStart configuration
                        "k_pos": 10000,
                        "k_neg": 10000,
                        "distance": CosineDistance,
                        "normalizers": [StandardScalerNormalizer()],
                        "positive_min_threshold": 0,
                        "max_alpha": 0.05,
                        "min_alpha": -0.02,
                        "beta_scale": 0,
                        "exclude": "sst2|warmstart",
                    },
                ),
            ),
        )
    ]

    for pairs in grouped_experiment_pairs_k_0:
        for experiment_pair in pairs:
            num_experiments = len(experiment_pair)

            # Get CPU core lists for each experiment
            cpu_lists = evenly_split_cpus(num_experiments)

            processes = []
            for idx, (
                id,
                dataset,
                name,
                device_id,
                bot_token,
                configuration,
            ) in enumerate(experiment_pair):
                cpu_list = cpu_lists[idx]
                p = Process(
                    target=warmstart_search,
                    args=(
                        id,
                        dataset,
                        name,
                        device_id,
                        cpu_list,
                        random_seed,
                        configuration,
                        bot_token,
                    ),
                )

                print(f"Experiment ID: {id}")
                print(f"Dataset: {name}")
                print(f"Device: {device_id}")
                print("WarmStart Configuration:")
                for key, value in configuration.items():
                    print(f"  {key}: {value}")
                print("-" * 40)

                processes.append(p)
                p.start()

            for p in processes:
                p.join()

def run_experiment(experiment, device_id, cpu_list, bot_token):
    print(f"Experiment ID: {experiment['id']}")
    print(f"Dataset: {experiment['dataset_name']}")
    print(f"Device: {device_id}")
    
    if experiment['warmstart'] is not None:
        print("WarmStart Configuration:")
        for key, value in experiment['warmstart'].items():
            print(f"  {key}: {value}")
        print("-" * 40)
    
    target_function = warmstart_search if experiment["multiobjective"] else warmstart_search_single_objective
    
    p = Process(
        target=target_function,
        args=(
            f"{experiment['id']}-rd:{experiment['seed']}",
            experiment["dataset"],
            experiment["dataset_name"],
            device_id,
            cpu_list,
            experiment["seed"],
            experiment["warmstart"],
            bot_token,
        ),
    )
    return p

def run_experiment_batch(batch, available_gpus, cpu_division, bot_tokens):
    processes = []
    for i, experiment in enumerate(batch):
        device_id = available_gpus[i]
        cpu_list = cpu_division[i]
        p = run_experiment(experiment, device_id, cpu_list, bot_tokens[i])
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

def main():
    # Initialize experiments list
    # seeds = [42, 123, 2024]
    seeds = [101, 707, 2029]
    available_gpus = [0, 1]
    bots = ["6759026182:AAEpd8vczWkLLaAoyPeCHezYAqH5vq9D9jg", "6425450979:AAF4Mic12nAWYlfiMNkCTRB0ZzcgaIegd7M"]
    
    multiseed_experiments = [
        # {
        #     "id": "9",
        #     "dataset": liar,
        #     "dataset_name": "liar",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": None,
        # },
        {
            "id": "(f-pos cos k=0.5)",
            "dataset": liar,
            "dataset_name": "liar",
            "multiobjective": False,
            "seed": None,
            "warmstart": {  # WarmStart configuration
                "k_pos": 10000,
                "k_neg": 0,
                "distance": CosineDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0.5,
                "f1_weight": 1.0,
                "evaluation_time_weight": 0.0,
                "exclude": "liar|warmstart",
            },
        },
        {
            "id": "(f-pos + a-neg)",
            "dataset": liar,
            "dataset_name": "liar",
            "multiobjective": False,
            "seed": None,
            "warmstart": {  # WarmStart configuration
                "k_pos": 10000,
                "k_neg": 10000,
                "distance": EuclideanDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "adaptative_negative_alpha_limit": -1,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0,
                "f1_weight": 1.0,
                "evaluation_time_weight": 0.0,
                "exclude": "liar|warmstart",
            },
        },
        # {
        #     "id": "(f-pos f-neg)",
        #     "dataset": liar,
        #     "dataset_name": "liar",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0,
        #         "f1_weight": 1.0,
        #         "evaluation_time_weight": 0.0,
        #         "exclude": "liar|warmstart",
        #     },
        # },
        
        # START SST2 SINGLE OBJECTIVE EXPERIMENTS
        # {
        #     "id": "9",
        #     "dataset": sst2,
        #     "dataset_name": "sst2",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": None,
        # },
        # {
        #     "id": "(f-pos + a-neg)",
        #     "dataset": sst2,
        #     "dataset_name": "sst2",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0,
        #         "f1_weight": 1.0,
        #         "evaluation_time_weight": 0.0,
        #         "exclude": "sst2|warmstart",
        #     },
        # },
        {
            "id": "(f-pos cos k=0.5)",
            "dataset": sst2,
            "dataset_name": "sst2",
            "multiobjective": False,
            "seed": None,
            "warmstart": {  # WarmStart configuration
                "k_pos": 10000,
                "k_neg": 0,
                "distance": CosineDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0.5,
                "f1_weight": 1.0,
                "evaluation_time_weight": 0.0,
                "exclude": "sst2|warmstart",
            },
        },
        {
            "id": "(f-pos f-neg)",
            "dataset": sst2,
            "dataset_name": "sst2",
            "multiobjective": False,
            "seed": None,
            "warmstart": {  # WarmStart configuration
                "k_pos": 10000,
                "k_neg": 10000,
                "distance": CosineDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0,
                "f1_weight": 1.0,
                "evaluation_time_weight": 0.0,
                "exclude": "sst2|warmstart",
            },
        },
        
        # START MELD SINGLE OBJECTIVE EXPERIMENTS
        # {
        #     "id": "9",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": None,
        # },
        # {
        #     "id": "(f-pos + a-neg)",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0,
        #         "f1_weight": 1.0,
        #         "evaluation_time_weight": 0.0,
        #         "exclude": "meld|warmstart",
        #     },
        # },
        # {
        #     "id": "(f-pos cos k=0.5)",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 0,
        #         "distance": CosineDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5,
        #         "f1_weight": 1.0,
        #         "evaluation_time_weight": 0.0,
        #         "exclude": "meld|warmstart",
        #     },
        # },
        # {
        #     "id": "(f-pos f-neg)",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": False,
        #     "seed": None,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 10000,
        #         "distance": CosineDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0,
        #         "f1_weight": 1.0,
        #         "evaluation_time_weight": 0.0,
        #         "exclude": "meld|warmstart",
        #     },
        # },
        # END SINGLE OBJECTIVE EXPERIMENTS
    ]
    
    fixed_seed_experiments = [
        # # START MULTI OBJECTIVE EXPERIMENTS LIAR
        # # { # low median
        # #     "id": "(no-pos a-neg euc k=0.5)[F1+Time][linear_front]",
        # #     "dataset": liar,
        # #     "dataset_name": "liar",
        # #     "multiobjective": True,
        # #     "seed": 42,
        # #     "warmstart": {  # WarmStart configuration
        # #         "k_pos": 0,
        # #         "k_neg": 10000,
        # #         "distance": EuclideanDistance,
        # #         "normalizers": [StandardScalerNormalizer()],
        # #         "positive_min_threshold": 0,
        # #         "adaptative_negative_alpha_limit": -1,
        # #         "max_alpha": 0.05,
        # #         "min_alpha": -0.02,
        # #         "beta_scale": 0.5, # k=0.5
        # #         "f1_weight": 0.5,
        # #         "evaluation_time_weight": 0.5,
        # #         "exclude": "liar|warmstart",
        # #         "utility_function": "linear_front",
        # #     },
        # # },
        # { # low max
        #     "id": "(no-pos a-neg)[F1+Time][weighted_sum]",
        #     "dataset": liar,
        #     "dataset_name": "liar",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0, # no distance
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "liar|warmstart",
        #         "utility_function": "weighted_sum",
        #     },
        # },
        # # { # mod median
        # #     "id": "(a-pos a-neg euc k=0.5)[F1+Time][weighted_sum]",
        # #     "dataset": liar,
        # #     "dataset_name": "liar",
        # #     "multiobjective": True,
        # #     "seed": 42,
        # #     "warmstart": {  # WarmStart configuration
        # #         "k_pos": 10000,
        # #         "k_neg": 10000,
        # #         "distance": EuclideanDistance,
        # #         "normalizers": [StandardScalerNormalizer()],
        # #         "positive_min_threshold": 0,
        # #         "adaptative_positive_alpha_limit": 1,
        # #         "adaptative_negative_alpha_limit": -1,
        # #         "max_alpha": 0.05,
        # #         "min_alpha": -0.02,
        # #         "beta_scale": 0.5, # k=0.5
        # #         "f1_weight": 0.5,
        # #         "evaluation_time_weight": 0.5,
        # #         "exclude": "liar|warmstart",
        # #         "utility_function": "weighted_sum",
        # #     },
        # # },
        # { # mod max
        #     "id": "(no-pos f-neg cos k=0.5)[F1+Time][weighted_sum]",
        #     "dataset": liar,
        #     "dataset_name": "liar",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": CosineDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 1, # k=1
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "liar|warmstart",
        #         "utility_function": "weighted_sum",
        #     },
        # },
        # # { # high median
        # #     "id": "(no-pos f-neg euc k=0.5)[F1+Time][linear_front]",
        # #     "dataset": liar,
        # #     "dataset_name": "liar",
        # #     "multiobjective": True,
        # #     "seed": 42,
        # #     "warmstart": {  # WarmStart configuration
        # #         "k_pos": 0,
        # #         "k_neg": 10000,
        # #         "distance": EuclideanDistance,
        # #         "normalizers": [StandardScalerNormalizer()],
        # #         "positive_min_threshold": 0,
        # #         "max_alpha": 0.05,
        # #         "min_alpha": -0.02,
        # #         "beta_scale": 0.5, # k=0.5
        # #         "f1_weight": 0.5,
        # #         "evaluation_time_weight": 0.5,
        # #         "exclude": "liar|warmstart",
        # #         "utility_function": "linear_front",
        # #     },
        # # },
        # { # high max
        #     "id": "(f-pos f-neg)[F1+Time][weighted_sum]",
        #     "dataset": liar,
        #     "dataset_name": "liar",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0, # no distance
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "liar|warmstart",
        #         "utility_function": "weighted_sum",
        #     },
        # },
        # # END MULTI OBJECTIVE EXPERIMENTS LIAR
        
        # # START MULTI OBJECTIVE EXPERIMENTS SST2
        # # { # low median
        # #     "id": "(no-pos a-neg euc k=0.5)[F1+Time][linear_front]",
        # #     "dataset": sst2,
        # #     "dataset_name": "sst2",
        # #     "multiobjective": True,
        # #     "seed": 42,
        # #     "warmstart": {  # WarmStart configuration
        # #         "k_pos": 0,
        # #         "k_neg": 10000,
        # #         "distance": EuclideanDistance,
        # #         "normalizers": [StandardScalerNormalizer()],
        # #         "positive_min_threshold": 0,
        # #         "adaptative_negative_alpha_limit": -1,
        # #         "max_alpha": 0.05,
        # #         "min_alpha": -0.02,
        # #         "beta_scale": 0.5,  # k=0.5
        # #         "f1_weight": 0.5,
        # #         "evaluation_time_weight": 0.5,
        # #         "exclude": "sst2|warmstart",
        # #         "utility_function": "linear_front",
        # #     },
        # # },
        # { # low max
        #     "id": "(no-pos a-neg)[F1+Time][weighted_sum]",
        #     "dataset": sst2,
        #     "dataset_name": "sst2",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0,  # no distance
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "sst2|warmstart",
        #         "utility_function": "weighted_sum",
        #     },
        # },
        # # { # mod median
        # #     "id": "(a-pos no-neg euc k=0.5)[F1+Time][linear_front]",
        # #     "dataset": sst2,
        # #     "dataset_name": "sst2",
        # #     "multiobjective": True,
        # #     "seed": 42,
        # #     "warmstart": {  # WarmStart configuration
        # #         "k_pos": 10000,
        # #         "k_neg": 0,
        # #         "distance": EuclideanDistance,
        # #         "normalizers": [StandardScalerNormalizer()],
        # #         "positive_min_threshold": 0,
        # #         "adaptative_positive_alpha_limit": 1,
        # #         "max_alpha": 0.05,
        # #         "min_alpha": -0.02,
        # #         "beta_scale": 0.5,  # k=0.5
        # #         "f1_weight": 0.5,
        # #         "evaluation_time_weight": 0.5,
        # #         "exclude": "sst2|warmstart",
        # #         "utility_function": "linear_front",
        # #     },
        # # },
        # { # mod max
        #     "id": "(f-pos a-neg)[F1+Time][logarithmic_front]",
        #     "dataset": sst2,
        #     "dataset_name": "sst2",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0,  # no distance
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "sst2|warmstart",
        #         "utility_function": "logarithmic_front",
        #     },
        # },
        # # { # high median
        # #     "id": "(a-pos f-neg euc k=0.5)[F1+Time][weighted_sum]",
        # #     "dataset": sst2,
        # #     "dataset_name": "sst2",
        # #     "multiobjective": True,
        # #     "seed": 42,
        # #     "warmstart": {  # WarmStart configuration
        # #         "k_pos": 10000,
        # #         "k_neg": 10000,
        # #         "distance": EuclideanDistance,
        # #         "normalizers": [StandardScalerNormalizer()],
        # #         "positive_min_threshold": 0,
        # #         "adaptative_positive_alpha_limit": 1,
        # #         "max_alpha": 0.05,
        # #         "min_alpha": -0.02,
        # #         "beta_scale": 0.5, # k=0.5
        # #         "f1_weight": 0.5,
        # #         "evaluation_time_weight": 0.5,
        # #         "exclude": "sst2|warmstart",
        # #         "utility_function": "weighted_sum",
        # #     },
        # # },
        # { # high max
        #     "id": "(no-pos f-neg cos k=0.5)[F1+Time][weighted_sum]",
        #     "dataset": sst2,
        #     "dataset_name": "sst2",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": CosineDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "sst2|warmstart",
        #         "utility_function": "weighted_sum",
        #     },
        # },
        
        # START MULTI OBJECTIVE EXPERIMENTS MELD
        # { # low median
        #     "id": "(no-pos a-neg euc k=0.5)[F1+Time][linear_front]",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "meld|warmstart",
        #         "utility_function": "linear_front",
        #     },
        # },
        { # low max
            "id": "(no-pos a-neg)[F1+Time][weighted_sum]",
            "dataset": meld,
            "dataset_name": "meld",
            "multiobjective": True,
            "seed": 42,
            "warmstart": {  # WarmStart configuration
                "k_pos": 0,
                "k_neg": 10000,
                "distance": EuclideanDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "adaptative_negative_alpha_limit": -1,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0, # no distance
                "f1_weight": 0.5,
                "evaluation_time_weight": 0.5,
                "exclude": "meld|warmstart",
                "utility_function": "weighted_sum",
            },
        },
        # { # mod median
        #     "id": "(a-pos no-neg cos k=0.5)[F1+Time][linear_front]",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 0,
        #         "distance": CosineDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_positive_alpha_limit": 1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "meld|warmstart",
        #         "utility_function": "linear_front",
        #     },
        # },
        { # mod max
            "id": "(no-pos f-neg euc k=1)[F1+Time][weighted_sum]",
            "dataset": meld,
            "dataset_name": "meld",
            "multiobjective": True,
            "seed": 42,
            "warmstart": {  # WarmStart configuration
                "k_pos": 0,
                "k_neg": 10000,
                "distance": EuclideanDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 1, # k=1
                "f1_weight": 0.5,
                "evaluation_time_weight": 0.5,
                "exclude": "meld|warmstart",
                "utility_function": "weighted_sum",
            },
        },
        # { # high median
        #     "id": "(no-pos f-neg euc k=0.5)[F1+Time][linear_front]",
        #     "dataset": meld,
        #     "dataset_name": "meld",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "meld|warmstart",
        #         "utility_function": "linear_front",
        #     },
        # },
        { # high max
            "id": "(f-pos f-neg)[F1+Time][logarithmic_front]",
            "dataset": meld,
            "dataset_name": "meld",
            "multiobjective": True,
            "seed": 42,
            "warmstart": {  # WarmStart configuration
                "k_pos": 10000,
                "k_neg": 10000,
                "distance": EuclideanDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0, #no distance
                "f1_weight": 0.5,
                "evaluation_time_weight": 0.5,
                "exclude": "meld|warmstart",
                "utility_function": "logarithmic_front",
            },
        },
        
        # START MULTI OBJECTIVE EXPERIMENTS AG NEWS
        # { # low median
        #     "id": "(no-pos a-neg euc k=0.5)[F1+Time][linear_front]",
        #     "dataset": ag_news,
        #     "dataset_name": "ag_news",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_negative_alpha_limit": -1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "ag_news|warmstart",
        #         "utility_function": "linear_front",
        #     },
        # },
        { # low max
            "id": "(no-pos a-neg)[F1+Time][weighted_sum]",
            "dataset": ag_news,
            "dataset_name": "ag_news",
            "multiobjective": True,
            "seed": 42,
            "warmstart": {  # WarmStart configuration
                "k_pos": 0,
                "k_neg": 10000,
                "distance": EuclideanDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "adaptative_negative_alpha_limit": -1,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0, # no distance
                "f1_weight": 0.5,
                "evaluation_time_weight": 0.5,
                "exclude": "meld|warmstart",
                "utility_function": "weighted_sum",
            },
        },
        # { # mod median
        #     "id": "(a-pos + no-neg cos k=0.5)[F1+Time][linear_front]",
        #     "dataset": ag_news,
        #     "dataset_name": "ag_news",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 10000,
        #         "k_neg": 0,
        #         "distance": CosineDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "adaptative_positive_alpha_limit": 1,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "ag_news|warmstart",
        #         "utility_function": "linear_front",
        #     },
        # },
        { # mod max
            "id": "(f-pos + a-neg)[F1+Time][logarithmic_front]",
            "dataset": ag_news,
            "dataset_name": "ag_news",
            "multiobjective": True,
            "seed": 42,
            "warmstart": {  # WarmStart configuration
                "k_pos": 10000,
                "k_neg": 10000,
                "distance": CosineDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "adaptative_negative_alpha_limit": -1,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0, # no distance
                "f1_weight": 0.5,
                "evaluation_time_weight": 0.5,
                "exclude": "meld|warmstart",
                "utility_function": "logarithmic_front",
            },
        },
        # { # high median
        #     "id": "(no-pos f-neg euc k=0.5)[F1+Time][linear_front]",
        #     "dataset": ag_news,
        #     "dataset_name": "ag_news",
        #     "multiobjective": True,
        #     "seed": 42,
        #     "warmstart": {  # WarmStart configuration
        #         "k_pos": 0,
        #         "k_neg": 10000,
        #         "distance": EuclideanDistance,
        #         "normalizers": [StandardScalerNormalizer()],
        #         "positive_min_threshold": 0,
        #         "max_alpha": 0.05,
        #         "min_alpha": -0.02,
        #         "beta_scale": 0.5, # k=0.5
        #         "f1_weight": 0.5,
        #         "evaluation_time_weight": 0.5,
        #         "exclude": "ag_news|warmstart",
        #         "utility_function": "linear_front",
        #     },
        # },
        { # high max
            "id": "(no-pos f-neg)[F1+Time][weighted_sum]",
            "dataset": ag_news,
            "dataset_name": "ag_news",
            "multiobjective": True,
            "seed": 42,
            "warmstart": {  # WarmStart configuration
                "k_pos": 0,
                "k_neg": 10000,
                "distance": EuclideanDistance,
                "normalizers": [StandardScalerNormalizer()],
                "positive_min_threshold": 0,
                "max_alpha": 0.05,
                "min_alpha": -0.02,
                "beta_scale": 0, #no distance
                "f1_weight": 0.5,
                "evaluation_time_weight": 0.5,
                "exclude": "meld|warmstart",
                "utility_function": "weighted_sum",
            },
        }
    ]
    
    converted_multiseed_experiments = []
    for exp in multiseed_experiments:
        if (exp["seed"] is not None):
            converted_multiseed_experiments.append(exp)
        else:
            for seed in seeds:
                nexp = exp.copy()
                nexp["seed"] = seed
                converted_multiseed_experiments.append(nexp)
            
    total_experiments = fixed_seed_experiments#fixed_seed_experiments + converted_multiseed_experiments
    experiment_batch_size = len(available_gpus)
    cpu_division = evenly_split_cpus(experiment_batch_size)
    total_batches = math.ceil(len(total_experiments) / experiment_batch_size)
    
    for batch_index in range(total_batches):
        start_index = batch_index * experiment_batch_size
        end_index = min((batch_index + 1) * experiment_batch_size, len(total_experiments))
        current_batch = total_experiments[start_index:end_index]
        run_experiment_batch(current_batch, available_gpus, cpu_division, bots)
    
if __name__ == "__main__":
    # run_liar_metalearning_test()
    main()
