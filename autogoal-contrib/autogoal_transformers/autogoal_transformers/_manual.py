import warnings

from autogoal.kb import (
    Label,
    Seq,
    Supervised,
    Word,
    Prompt,
    GeneratedText,
    Sentence,
    MatrixContinuousDense,
    VectorCategorical,
    VectorDiscrete,
    Document,
)
from autogoal.kb._algorithm import _make_list_args_and_kwargs
from autogoal.kb import algorithm, AlgorithmBase, VectorContinuous
from autogoal.grammar import (
    DiscreteValue,
    CategoricalValue,
    BooleanValue,
)
from autogoal.utils import nice_repr
from autogoal_transformers._builder import (
    TransformersWrapper,
)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from autogoal.utils._process import is_cuda_multiprocessing_enabled
import textwrap
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
)
from autogoal_transformers._utils import SimpleTextDataset, Text2TextDataset, safe_model_from_pretrained
from peft import get_peft_model, LoraConfig, TaskType
import os

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)


@nice_repr
class SeqPretrainedTokenClassifier(AlgorithmBase):
    def __init__(
        self,
        pretrained_token_classifier: algorithm(Seq[Word], Supervised[Seq[Label]], Seq[Label]),  # type: ignore
    ) -> None:
        super().__init__()
        self.inner = pretrained_token_classifier

    def run(self, X: Seq[Seq[Word]], y: Supervised[Seq[Seq[Label]]]) -> Seq[Seq[Label]]:
        args_kwargs = _make_list_args_and_kwargs(X, y)
        return [self.inner.run(*t.args, **t.kwargs) for t in args_kwargs]


@nice_repr
class TGenerationBasedPretrainedEmbedder(AlgorithmBase):
    def __init__(
        self,
        pretrained_text_generator: algorithm(Seq[Prompt], Seq[GeneratedText]),  # type: ignore
    ) -> None:
        super().__init__()
        self.pretrained_text_generator = pretrained_text_generator
        self.batch_size = 128  # self.pretrained_text_generator.batch_size
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        self.device = torch.cuda._get_device(self.device)
        device_name = torch.cuda.get_device_name(self.device)

    def run(self, X: Seq[Sentence]) -> MatrixContinuousDense:
        self.pretrained_text_generator.init_model()
        embeddings_matrix = []
        for i in tqdm(range(0, len(X), self.batch_size)):
            batch_sentences = X[i : i + self.batch_size]
            inputs = self.pretrained_text_generator.tokenizer.batch_encode_plus(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                model = (
                    self.pretrained_text_generator.model.get_encoder()
                    if self.pretrained_text_generator.is_encoder_decoder
                    else self.pretrained_text_generator.model
                )

                outputs = model(**inputs, output_hidden_states=True)
                # outputs[0] contains the last hidden state
                batch_embeddings = (
                    outputs.hidden_states[-1].mean(dim=1).to("cpu").numpy()
                )

            embeddings_matrix.extend(batch_embeddings)
            del inputs, outputs
            torch.cuda.empty_cache()
        return np.vstack(embeddings_matrix)


@nice_repr
class CARPClassifier(TransformersWrapper):
    def __init__(
        self,
        few_shots_amount: CategoricalValue(2, 4, 8, 16, 32, 64, 128),  # type: ignore
        training_examples_selection_method: CategoricalValue("random"),  # type: ignore
        pretrained_text_generator: algorithm(Seq[Prompt], Seq[GeneratedText]),  # type: ignore
    ) -> None:
        super().__init__()
        self.pretrained_text_generator = pretrained_text_generator
        self.batch_size = self.pretrained_text_generator.batch_size
        self.few_shots_amount = few_shots_amount
        self.training_examples_selection_method = training_examples_selection_method
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        self.device = torch.cuda._get_device(self.device)
        device_name = torch.cuda.get_device_name(self.device)

    def _train(self, X, y):
        self.pretrained_text_generator.init_model()
        self.pretrained_text_generator.max_gen_seq_length = 200
        self.pretrained_text_generator.temperature = 1

        self.augmented_data = self.augment_data(X, y)
        return y

    def _eval(self, X, y) -> VectorCategorical:
        self.pretrained_text_generator.init_model()

        base_prompt = textwrap.dedent(
            f"""
            This is a text classifier. Only respond with the text to complete and nothing more at all.
            
            List CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the class determination of the input. 
            Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the class determination. 
            Finally, based on clues, the reasoning and the input, categorize the overall classof input as one of the following: {unique_labels_text}.
            
            Your answer should be the CLUES, REASONING and later the LABEL for the target. Make sure you base your response on the examples below.
            """
        )

        unique_labels_text = ", ".join(self.unique_labels)
        augmented_prompts = []
        for i in range(len(X)):
            training_examples = []
            if self.training_examples_selection_method == "random":
                training_examples = np.random.choice(
                    range(len(self.augmented_data)), self.few_shots_amount
                )

            training_examples_text = "\n".join(
                [self.augmented_data[i]["training_example"] for i in training_examples]
            )
            augmented_prompts.append(
                base_prompt
                + textwrap.dedent(
                    f"""
                {training_examples_text}
                
                target
                INPUT: {X[i]}
                CLUES:
                REASONING:
                LABEL:
                """
                )
            )

        generated_text = self.pretrained_text_generator.run(augmented_prompts)
        labels_pattern = "|".join(map(re.escape, self.unique_labels))
        pattern = f"{labels_pattern}"

        results = []
        for text in generated_text:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                results.append(matches[-1])
            else:
                results.append(np.random.choice(self.unique_labels))

    def augment_data(self, X, y):
        unique_labels = np.unique(y)
        unique_labels_text = ", ".join(unique_labels)

        # prepare clue prompts
        augmented_data = []
        for sequence, label in zip(X, y):
            augmented_data.append(
                {
                    "sequence": sequence,
                    "label": label,
                    "clue_prompt": textwrap.dedent(
                        f"""
                    Only respond with the text to complete and nothing more at all.
                    This is a generic text classifier. The possible classes are: {unique_labels_text}.
                    
                    Complete the CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the label determination of the input (limit to 15 words).
                    Your response should be a list of words or phrases separated by commas and should be right after "CLUES:".
                    
                    INPUT: {sequence}
                    GOLD LABEL: {label}
                    CLUES: 
                """
                    ),
                }
            )

        clues = self.pretrained_text_generator.run(
            [x["clue_prompt"] for x in augmented_data]
        )
        for i in range(len(augmented_data)):
            gen_clues = clues[i]
            prompt = augmented_data[i]["clue_prompt"]
            answer_index = clues[i].find(prompt) + len(prompt)

            if answer_index >= 0:
                gen_clues = clues[i][answer_index:]

            augmented_data[i]["clues"] = gen_clues
            augmented_data[i]["clue_prompt"] = None
            augmented_data[i]["reasoning_prompt"] = textwrap.dedent(
                f"""
                    Only respond with the text to complete and nothing more at all.
                    This is a generic text classifier. The possible classes are: {unique_labels_text}.
                    
                    Based on the input and clues, articulate the diagnostic reasoning process that supports the class (label) determination of the input.
                    Your response should be a reasoning text and should be right after "REASONING:".
                    
                    INPUT: {sequence}
                    LABEL: {label}
                    CLUES: {clues[i]}
                    REASONING:
                """
            )

        reasonings = self.pretrained_text_generator.run(
            [x["reasoning_prompt"] for x in augmented_data]
        )
        for i in range(len(augmented_data)):
            gen_reasoning = reasonings[i]
            prompt = augmented_data[i]["reasoning_prompt"]
            answer_index = clues[i].find(prompt) + len(prompt)

            if answer_index >= 0:
                gen_reasoning = reasonings[i][answer_index:]

            augmented_data[i]["reasonings"] = gen_reasoning
            augmented_data[i]["reasoning_prompt"] = None
            augmented_data[i]["training_example"] = textwrap.dedent(
                f"""

                example {i}
                INPUT: {augmented_data[i]['sequence']} 
                CLUES: {augmented_data[i]['clues']} 
                REASONING: {augmented_data[i]['reasonings']}
                LABEL: {augmented_data[i]['label']}
                """
            )

        self.augmented_data = augmented_data
        self.unique_labels = unique_labels
        return augmented_data

    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)


@nice_repr
class GenerativeClassifier(TransformersWrapper):
    def __init__(
        self,
        zero_shot: BooleanValue(),  # type: ignore
        few_shots_amount: CategoricalValue(16, 32, 64, 128),  # type: ignore
        training_examples_selection_method: CategoricalValue("random"),  # type: ignore
        pretrained_text_generator: algorithm(Seq[Prompt], Seq[GeneratedText]),  # type: ignore
    ) -> None:
        super().__init__()
        self.pretrained_text_generator = pretrained_text_generator
        self.batch_size = self.pretrained_text_generator.batch_size
        self.zero_shot = zero_shot
        self.few_shots_amount = few_shots_amount
        self.training_examples_selection_method = training_examples_selection_method
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        self.device = torch.cuda._get_device(self.device)
        device_name = torch.cuda.get_device_name(self.device)

    def _train(self, X, y):
        self.pretrained_text_generator.init_model()
        self.pretrained_text_generator.max_gen_seq_length = 200
        self.pretrained_text_generator.temperature = 1
        self.unique_labels = np.unique(y)
        self.unique_labels_text = ", ".join(self.unique_labels)

        if self.zero_shot:
            return y

        self.augmented_data = list(zip(X, y))
        return y

    def _eval(self, X, y) -> VectorCategorical:
        self.pretrained_text_generator.init_model()
        base_prompt = textwrap.dedent(
            f"""
            This is a text classifier. Only respond with the target class to predict. Follow the next steps for arriving to a target LABEL.
            Categorize the target class of input as one of the following: {self.unique_labels_text}.
            
            Make sure you base your response on the examples below. RESPOND ONLY WITH THE LABEL!!!.
            """
        )

        augmented_prompts = []
        if not self.zero_shot:
            for i in range(len(X)):
                training_examples = []
                if self.training_examples_selection_method == "random":
                    training_examples = np.random.choice(
                        range(len(self.augmented_data)), self.few_shots_amount
                    )

                training_examples_text = "\n".join(
                    [
                        f"""
                    example {i}
                    INPUT: {self.augmented_data[i][0]}
                    LABEL: {self.augmented_data[i][1]}
                    """
                        for i in training_examples
                    ]
                )

                nprompt = base_prompt + textwrap.dedent(
                    f"""
                    {training_examples_text}
                    
                    target
                    INPUT: {X[i]}
                    LABEL:
                    """
                )
                augmented_prompts.append(nprompt)
            else:
                augmented_prompts = [
                    base_prompt
                    + textwrap.dedent(
                        f"""
                    target
                    INPUT: {X[i]}
                    LABEL:
                    """
                    )
                    for i in range(len(X))
                ]

        generated_text = self.pretrained_text_generator.run(augmented_prompts)
        labels_pattern = "|".join(map(re.escape, self.unique_labels))
        pattern = f"{labels_pattern}"

        results = []
        for text in generated_text:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                results.append(matches[-1])
            else:
                results.append(np.random.choice(self.unique_labels))
        return results

    def run(
        self, X: Seq[Sentence], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return TransformersWrapper.run(self, X, y)


@nice_repr
class DocumentEmbedder(AlgorithmBase):
    def __init__(
        self,
        seq_embedder: algorithm(Seq[Sentence], MatrixContinuousDense, exceptions=["DocumentEmbedder"]),  # type: ignore
        sent_tokenizer: algorithm(Document, Seq[Sentence]),  # type: ignore
        pooling: CategoricalValue("mean", "max", "rms"),  # type: ignore
        normalization_strategy: CategoricalValue("l2", "l1", "min-max", "z-score", "none"),  # type: ignore
    ) -> None:
        super().__init__()
        self.seq_embedder = seq_embedder
        self.sent_tokenizer = sent_tokenizer
        self.pooling = pooling
        self.normalization_strategy = normalization_strategy
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        self.device = torch.cuda._get_device(self.device)
        device_name = torch.cuda.get_device_name(self.device)

    def run(self, X: Seq[Document]) -> MatrixContinuousDense:
        all_sentences = []  # To store all sentences from all documents
        doc_to_sent_indices = []  # To track which sentences belong to which document

        for doc in X:
            sentences = self.sent_tokenizer.run(
                doc
            )  # Assuming this returns a list of sentences
            all_sentences.extend(sentences)
            doc_to_sent_indices.append(len(sentences))

        # Step 2: Embed sentences
        sentence_embeddings = self.seq_embedder.run(all_sentences)

        # Step 3: Group embeddings by document and apply pooling
        doc_embeddings = []
        start_idx = 0
        for num_sentences in doc_to_sent_indices:
            end_idx = start_idx + num_sentences
            doc_sent_embeddings = sentence_embeddings[start_idx:end_idx]

            doc_embeddings.append(self.pool(doc_sent_embeddings))
            start_idx = end_idx

        if self.normalization_strategy != "none":
            doc_embeddings = self.normalize(torch.stack(doc_embeddings))

        return doc_embeddings

    def pool(self, doc_sent_embeddings):
        doc_sent_embeddings_tensors = [
            torch.tensor(emb, dtype=torch.float) if isinstance(emb, np.ndarray) else emb
            for emb in doc_sent_embeddings
        ]
        stacked_embeddings = torch.stack(doc_sent_embeddings_tensors).to(self.device)

        if self.pooling == "mean":
            doc_embedding = torch.mean(stacked_embeddings, dim=0)
        elif self.pooling == "max":
            doc_embedding, _ = torch.max(stacked_embeddings, dim=0)
        elif self.pooling == "rms":
            doc_embedding = torch.sqrt(torch.mean(stacked_embeddings**2, dim=0))
        else:
            raise ValueError("Unsupported pooling method")
        return doc_embedding.to("cpu")

    def normalize(self, embeddings):
        if self.normalization_strategy == "l2":
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        elif self.normalization_strategy == "l1":
            normalized_embeddings = F.normalize(embeddings, p=1, dim=1)
        elif self.normalization_strategy == "min-max":
            min_val = embeddings.min(dim=1, keepdim=True)[0]
            max_val = embeddings.max(dim=1, keepdim=True)[0]
            normalized_embeddings = (embeddings - min_val) / (max_val - min_val)
        elif self.normalization_strategy == "z-score":
            mean = embeddings.mean(dim=1, keepdim=True)
            std = embeddings.std(dim=1, keepdim=True)
            normalized_embeddings = (embeddings - mean) / std
        elif self.normalization_strategy == "none":
            normalized_embeddings = embeddings  # No normalization applied
        else:
            raise ValueError(
                f"Unknown normalization strategy: {self.normalization_strategy}"
            )
        return normalized_embeddings.to("cpu")


@nice_repr
class FineTunerBase(AlgorithmBase):
    def __init__(
        self,
    ):
        self._mode = "train"
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        self.device = torch.cuda._get_device(self.device)
        device_name = torch.cuda.get_device_name(self.device)

        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def count_trainable_parameters(self):

        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def init_model(self, inner_model, num_labels):
        inner_model_name = inner_model.name
        print(f"Initializing model: {inner_model_name}")
        assert isinstance(inner_model_name, str), "Model name must be a string"
        assert (
            isinstance(num_labels, int) and num_labels > 0
        ), "num_labels must be a positive integer"

        self.config = AutoConfig.from_pretrained(
            inner_model_name,
            num_labels=num_labels,
            hidden_dropout_prob=(
                self.dropout_rate if hasattr(self, "dropout_rate") else 0
            ),
            attention_probs_dropout_prob=(
                self.dropout_rate if hasattr(self, "dropout_rate") else 0
            ),
            trust_remote_code=True,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                inner_model_name, use_fast=True
            )
        except Exception as e:
            print(f"Error loading tokenizer for model '{inner_model_name}': {e}")
            raise e

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                inner_model_name,
                config=self.config,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading model for '{inner_model_name}': {e}")
            raise e

        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            print("No padding token. Adding EOS as PAD token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        max_model_length = self.tokenizer.model_max_length
        if max_model_length < self.max_length:
            print(
                f"Model max length is {max_model_length} and the provided max length is {self.max_length}. Using the tokenizer max length."
            )
            self.max_length = min(self.max_length, max_model_length)

    @staticmethod
    def get_num_workers(num_workers_option):
        import math
        import os

        total_cpus = os.cpu_count()
        if num_workers_option == "all":
            return total_cpus
        elif num_workers_option == "3/4":
            return max(1, math.floor(0.75 * total_cpus))
        elif num_workers_option == "half":
            return max(1, math.floor(0.5 * total_cpus))
        elif num_workers_option == "1/4":
            return max(1, math.floor(0.25 * total_cpus))
        else:
            # Default to 1 if unrecognized option
            return 0

    def setup_optimizer(self):
        if self.optimizer == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    def setup_scheduler(self, optimizer, total_steps):
        if self.lr_scheduler == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.lr_scheduler == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.lr_scheduler == "constant":
            return get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=self.warmup_steps
            )
        elif self.lr_scheduler == "cosine_with_restarts":
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
                num_cycles=1,
            )
        elif self.lr_scheduler == "polynomial":
            return get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
                power=1.0,
            )

    def finetune(self, X, y):
        pass

    def predict(self, X):
        dataset = SimpleTextDataset(X, None, self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key != "labels"
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        preds = self._postprocess_output(preds)
        return preds

    def _preprocess_input(self, X, y):
        return X, y

    def _postprocess_output(self, y):
        return y

    def run(self, X: Seq[Sentence], y: Supervised[VectorDiscrete]) -> VectorDiscrete:  # type: ignore
        if self._mode == "train":
            return self.finetune(X, y)
        else:
            return self.predict(X)


@nice_repr
class FineTuneLLMEmbeddingClassifier(FineTunerBase):
    def __init__(
        self,
        inner_model: algorithm(*[Word, VectorContinuous], include=["transformer"]),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        # New parameters to control features
        # use_early_stopping: BooleanValue(),  # type: ignore
        # early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        class_weighted_loss: BooleanValue(),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
    ):
        self.model = None
        self.tokenizer = None
        self.inner_model = inner_model
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler

        # New parameters for added features
        self.use_early_stopping = True
        self.early_stopping_patience = 2
        self.early_stopping_delta = early_stopping_delta
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.class_weighted_loss = class_weighted_loss
        self.num_workers = num_workers
        super().__init__()

    def finetune(self, X, y):
        X, y = self._preprocess_input(X, y)

        num_labels = len(np.unique(y))
        self.init_model(self.inner_model, num_labels)

        # Handle class imbalance
        if self.class_weighted_loss:
            y_int = np.array(y, dtype=int)
            class_counts = np.bincount(y_int)
            class_weights = 1.0 / class_counts
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Create dataset and dataloader
        dataset = SimpleTextDataset(X, y, self.tokenizer, max_length=self.max_length)
        num_workers = self.get_num_workers(self.num_workers)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        optimizer = self.setup_optimizer()
        total_steps = len(dataloader) * self.epochs
        scheduler = self.setup_scheduler(optimizer, total_steps)

        print(f"trainable parameters: {self.count_trainable_parameters()}")

        # Initialize early stopping variables
        if self.use_early_stopping:
            epochs_no_improve = 0

        # Initialize mixed precision scaler
        if self.use_mixed_precision and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
            use_mixed_precision = True
        else:
            use_mixed_precision = False

        previous_loss = None
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                inputs = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key != "labels"
                }
                labels = batch["labels"].to(self.device)

                if use_mixed_precision:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(**inputs)
                        loss = loss_fn(outputs.logits, labels)
                else:
                    outputs = self.model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                loss = (
                    loss / self.gradient_accumulation_steps
                    if self.gradient_accumulation_steps > 1
                    else loss
                )

                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(dataloader):
                    # Gradient clipping
                    if self.use_gradient_clipping:
                        if use_mixed_precision:
                            scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clipping_max_norm,
                        )

                    if use_mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss}")

            # Early stopping based on training loss plateau
            if self.use_early_stopping:
                if previous_loss is not None:
                    loss_diff = previous_loss - avg_loss
                    if loss_diff < self.early_stopping_delta:
                        epochs_no_improve += 1
                        print(
                            f"Epoch {epoch+1}: Training loss did not improve by at least {self.early_stopping_delta}. ({epochs_no_improve}/{self.early_stopping_patience})"
                        )
                    else:
                        epochs_no_improve = 0
                        print(f"Epoch {epoch+1}: Training loss improved.")

                    if epochs_no_improve >= self.early_stopping_patience:
                        print(
                            "Early stopping triggered due to no improvement in training loss."
                        )
                        break
                else:
                    print(
                        f"Epoch {epoch+1}: First epoch, setting baseline training loss."
                    )

                previous_loss = avg_loss

        return y


@nice_repr
class PartialFineTuneLLMEmbeddingClassifier(FineTunerBase):
    def __init__(
        self,
        inner_model: algorithm(*[Word, VectorContinuous], include=["transformer"]),  # type: ignore
        num_trainable_layers: CategoricalValue(1, 2, 4, 8, 16, 32, 64),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        # New parameters to control features
        # use_early_stopping: BooleanValue(),  # type: ignore
        # early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        class_weighted_loss: BooleanValue(),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
    ):
        self.model = None
        self.tokenizer = None
        self.inner_model = inner_model
        self.num_trainable_layers = num_trainable_layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler

        # New parameters for added features
        self.use_early_stopping = True
        self.early_stopping_patience = 2
        self.early_stopping_delta = early_stopping_delta
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.class_weighted_loss = class_weighted_loss
        self.num_workers = num_workers
        super().__init__()

    def set_freezed_layers(self):
        """
        Freezes all layers initially and then unfreezes the specified number of layers
        from the end (top) of the model. Also ensures that the classifier layers are trainable.
        """
        # Group layers by their number
        layer_groups = self.group_layers_by_number()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier layers
        classifier_params = [
            p for n, p in self.model.named_parameters() if "classifier" in n
        ]
        for param in classifier_params:
            param.requires_grad = True

        # Unfreeze additional layers based on num_trainable_layers
        layers_unfreezed = 0
        total_layers = len(layer_groups)

        if total_layers == 0:
            print(
                "No layers matched the freezing pattern. All non-classifier parameters remain frozen."
            )
        else:
            for layer_num in sorted(layer_groups.keys(), reverse=True):
                if layers_unfreezed < self.num_trainable_layers:
                    for param in layer_groups[layer_num]:
                        param.requires_grad = True
                    layers_unfreezed += 1
                else:
                    break

        # If still need to unfreeze more layers, unfreeze non-layer parameters
        if layers_unfreezed < self.num_trainable_layers:
            non_layer_params = [
                p
                for n, p in self.model.named_parameters()
                if not re.search(r"\.layer\.\d+\.", n)
                and not re.search(r"\.(encoder|decoder)\.block\.\d+\.", n)
                and "classifier" not in n
            ]
            for param in non_layer_params:
                param.requires_grad = True
            print(
                f"Unfrozen additional {self.num_trainable_layers - layers_unfreezed} non-layer parameters."
            )

        # Verify that at least some parameters are trainable
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(
                "No trainable parameters found. Please check the layer freezing configuration."
            )

    def group_layers_by_number(self):
        """
        Group parameters by their layer number for different model architectures.
        Supports both BERT-like and T5-like layer naming conventions.
        """
        layer_groups = {}

        # Patterns for different models
        patterns = [
            re.compile(r"\.layer\.(\d+)\."),  # BERT-like
            re.compile(r"\.(encoder|decoder)\.block\.(\d+)\."),  # T5-like
        ]

        for name, param in self.model.named_parameters():
            for pattern in patterns:
                match = pattern.search(name)
                if match:
                    # For BERT-like patterns
                    if pattern.pattern == r"\.layer\.(\d+)\.":
                        layer_num = int(match.group(1))
                    # For T5-like patterns
                    else:
                        layer_num = int(match.group(2))

                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = []
                    layer_groups[layer_num].append(param)
                    break  # Stop checking other patterns if a match is found

        return layer_groups

    def finetune(self, X, y):
        num_labels = len(np.unique(y))
        self.init_model(self.inner_model, num_labels)
        self.set_freezed_layers()

        # Verify that there are trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found after freezing layers.")

        print(f"trainable parameters: {self.count_trainable_parameters()}")

        # Handle class imbalance
        if self.class_weighted_loss:
            y_int = np.array(y, dtype=int)
            class_counts = np.bincount(y_int)
            class_weights = 1.0 / class_counts
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Create dataset and dataloader
        dataset = SimpleTextDataset(X, y, self.tokenizer, max_length=self.max_length)
        num_workers = self.get_num_workers(self.num_workers)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        optimizer = self.setup_optimizer()
        total_steps = len(dataloader) * self.epochs
        scheduler = self.setup_scheduler(optimizer, total_steps)

        print(f"trainable parameters: {self.count_trainable_parameters()}")

        # Initialize early stopping variables
        if self.use_early_stopping:
            epochs_no_improve = 0

        # Initialize mixed precision scaler
        if self.use_mixed_precision and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
            use_mixed_precision = True
        else:
            use_mixed_precision = False

        previous_loss = None
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                inputs = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key != "labels"
                }
                labels = batch["labels"].to(self.device)

                if use_mixed_precision:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(**inputs)
                        loss = loss_fn(outputs.logits, labels)
                else:
                    outputs = self.model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                loss = (
                    loss / self.gradient_accumulation_steps
                    if self.gradient_accumulation_steps > 1
                    else loss
                )

                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(dataloader):
                    # Gradient clipping
                    if self.use_gradient_clipping:
                        if use_mixed_precision:
                            scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clipping_max_norm,
                        )
                        
                    if use_mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss}")

            # Early stopping based on training loss plateau
            if self.use_early_stopping:
                if previous_loss is not None:
                    loss_diff = previous_loss - avg_loss
                    if loss_diff < self.early_stopping_delta:
                        epochs_no_improve += 1
                        print(
                            f"Epoch {epoch+1}: Training loss did not improve by at least {self.early_stopping_delta}. ({epochs_no_improve}/{self.early_stopping_patience})"
                        )
                    else:
                        epochs_no_improve = 0
                        print(f"Epoch {epoch+1}: Training loss improved.")

                    if epochs_no_improve >= self.early_stopping_patience:
                        print(
                            "Early stopping triggered due to no improvement in training loss."
                        )
                        break
                else:
                    print(
                        f"Epoch {epoch+1}: First epoch, setting baseline training loss."
                    )

                previous_loss = avg_loss

        return y


@nice_repr
class LoraLLMEmbeddingClassifier(FineTunerBase):
    def __init__(
        self,
        inner_model: algorithm(*[Word, VectorContinuous], include=["transformer"]),  # type: ignore
        lora_r: DiscreteValue(1, 32),  # type: ignore
        lora_alpha: CategoricalValue(8, 16, 32, 64),  # type: ignore
        lora_dropout: CategoricalValue(0.0, 0.1, 0.2, 0.3),  # type: ignore
        lora_bias: CategoricalValue("none", "all", "lora_only"),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        model_save: CategoricalValue("classifier", "all"),  # type: ignore
        init_lora_weights: CategoricalValue("gaussian", "pissa", "loftq", None),  # type: ignore
        fan_in_fan_out: CategoricalValue(True, False),  # type: ignore
        # New parameters to control features
        # use_early_stopping: BooleanValue(),  # type: ignore
        # early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        class_weighted_loss: BooleanValue(),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
    ):
        self.model = None
        self.tokenizer = None
        self.inner_model = inner_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler
        self.model_save = model_save
        self.init_lora_weights = init_lora_weights
        self.fan_in_fan_out = fan_in_fan_out

        # New parameters for added features
        self.use_early_stopping = True
        self.early_stopping_patience = 2
        self.early_stopping_delta = early_stopping_delta
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.class_weighted_loss = class_weighted_loss
        self.num_workers = num_workers
        super().__init__()

    def set_lora_config(self, target_modules=None):
        try:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                target_modules=target_modules,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                modules_to_save=(
                    [self.model_save] if (self.model_save == "classifier") else None
                ),
                init_lora_weights=self.init_lora_weights,
                fan_in_fan_out=self.fan_in_fan_out,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        except Exception as e:
            if target_modules is not None:
                print(
                    f"Failed to set LORA target modules. Trying with automatic LORA target modules discovery."
                )
                raise e

            print(
                "Failed automatic LORA target modules discovery. Trying with in-house modules detection."
            )
            return self.set_lora_config(self.get_specific_layer_names())

    def print_model_layers(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape)

    def get_specific_layer_names(self):
        model = self.model
        # Create a list to store the layer names
        layer_names = []
        # Recursively visit all modules and submodules
        for name, module in model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    torch.nn.Embedding,
                    torch.nn.Conv2d,
                    transformers.pytorch_utils.Conv1D,
                ),
            ):
                names = name.split(".")
                # model-specific
                layer_names.append(names[0] if len(names) == 1 else names[-1])

        return layer_names

    def finetune(self, X, y):
        X, y = self._preprocess_input(X, y)
        num_labels = len(np.unique(y))

        self.init_model(self.inner_model, num_labels)

        self.set_lora_config()

        # Handle class imbalance
        if self.class_weighted_loss:
            y_int = np.array(y, dtype=int)
            class_counts = np.bincount(y_int)
            class_weights = 1.0 / class_counts
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        # Create dataset and dataloader
        dataset = SimpleTextDataset(X, y, self.tokenizer, max_length=self.max_length)

        num_workers = self.get_num_workers(self.num_workers)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        optimizer = self.setup_optimizer()

        total_steps = len(dataloader) * self.epochs

        scheduler = self.setup_scheduler(optimizer, total_steps)

        # Initialize early stopping variables
        if self.use_early_stopping:
            epochs_no_improve = 0

        # Initialize mixed precision scaler
        if self.use_mixed_precision and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
            use_mixed_precision = True
        else:
            use_mixed_precision = False

        previous_loss = None
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                inputs = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key != "labels"
                }
                labels = batch["labels"].to(self.device)
                if use_mixed_precision:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(**inputs)
                        loss = loss_fn(outputs.logits, labels)
                else:
                    outputs = self.model(**inputs)
                    loss = loss_fn(outputs.logits, labels)

                loss = (
                    loss / self.gradient_accumulation_steps
                    if self.gradient_accumulation_steps > 1
                    else loss
                )

                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == len(dataloader):
                    # Gradient clipping
                    if self.use_gradient_clipping:
                        if use_mixed_precision:
                            scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.gradient_clipping_max_norm,
                        )
                        
                    if use_mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss}")

            # Early stopping based on training loss plateau
            if self.use_early_stopping:
                if previous_loss is not None:
                    loss_diff = previous_loss - avg_loss
                    if loss_diff < self.early_stopping_delta:
                        epochs_no_improve += 1
                        print(
                            f"Epoch {epoch+1}: Training loss did not improve by at least {self.early_stopping_delta}. ({epochs_no_improve}/{self.early_stopping_patience})"
                        )
                    else:
                        epochs_no_improve = 0
                        print(f"Epoch {epoch+1}: Training loss improved.")

                    if epochs_no_improve >= self.early_stopping_patience:
                        print(
                            "Early stopping triggered due to no improvement in training loss."
                        )
                        break
                else:
                    print(
                        f"Epoch {epoch+1}: First epoch, setting baseline training loss."
                    )

                previous_loss = avg_loss

        return y


@nice_repr
class FineTuneGenLLMClassifier(FineTuneLLMEmbeddingClassifier):
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        # New parameters to control features
        # use_early_stopping: BooleanValue(),  # type: ignore
        # early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        class_weighted_loss: BooleanValue(),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
    ):
        super().__init__(
            inner_model,
            batch_size,
            max_length,
            learning_rate,
            epochs,
            warmup_steps,
            weight_decay,
            dropout_rate,
            optimizer,
            gradient_accumulation_steps,
            lr_scheduler,
            # New parameters to control features
            early_stopping_delta,
            use_mixed_precision,
            use_gradient_clipping,
            gradient_clipping_max_norm,
            class_weighted_loss,
            num_workers,
        )


@nice_repr
class PartialFineTuneGenLLMClassifier(PartialFineTuneLLMEmbeddingClassifier):
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        num_trainable_layers: CategoricalValue(1, 2, 4, 8, 16, 32, 64),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        dropout_rate: CategoricalValue(0.1, 0.2, 0.3, 0.4, 0.5),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        # New parameters to control features
        # use_early_stopping: BooleanValue(),  # type: ignore
        # early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        class_weighted_loss: BooleanValue(),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
    ):
        super().__init__(
            inner_model,
            num_trainable_layers,
            batch_size,
            max_length,
            learning_rate,
            epochs,
            warmup_steps,
            weight_decay,
            optimizer,
            dropout_rate,
            gradient_accumulation_steps,
            lr_scheduler,
            # New parameters to control features
            early_stopping_delta,
            use_mixed_precision,
            use_gradient_clipping,
            gradient_clipping_max_norm,
            class_weighted_loss,
            num_workers,
        )


@nice_repr
class LoraGenLLMClassifier(LoraLLMEmbeddingClassifier):
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        lora_r: DiscreteValue(1, 32),  # type: ignore
        lora_alpha: CategoricalValue(8, 16, 32, 64),  # type: ignore
        lora_dropout: CategoricalValue(0.0, 0.1, 0.2, 0.3),  # type: ignore
        lora_bias: CategoricalValue("none", "all"),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        model_save: CategoricalValue("classifier", "all"),  # type: ignore
        init_lora_weights: CategoricalValue("gaussian", "pissa", "loftq", None),  # type: ignore
        fan_in_fan_out: CategoricalValue(True, False),  # type: ignore
        # New parameters to control features
        # use_early_stopping: BooleanValue(),  # type: ignore
        # early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        class_weighted_loss: BooleanValue(),  # type: ignore
        num_workers: CategoricalValue("default", "3/4"),  # type: ignore
    ):
        super().__init__(
            inner_model,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_bias,
            batch_size,
            max_length,
            learning_rate,
            epochs,
            warmup_steps,
            weight_decay,
            optimizer,
            gradient_accumulation_steps,
            lr_scheduler,
            model_save,
            init_lora_weights,
            fan_in_fan_out,
            early_stopping_delta,
            use_mixed_precision,
            use_gradient_clipping,
            gradient_clipping_max_norm,
            class_weighted_loss,
            num_workers,
        )


# === GENERATIVE FINETUNING ===
@nice_repr
class FineTunerGenBase(AlgorithmBase):
    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
        verbose: BooleanValue() = True,  # type: ignore
    ):
        super().__init__()
        # Track mode for training vs. evaluation
        self._mode = "train"
        self.inner_model = inner_model
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_scheduler = lr_scheduler
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clipping_max_norm = gradient_clipping_max_norm
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.num_workers = num_workers
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.is_encoder_decoder = None
        self.device = torch.cuda.current_device() if torch.cuda.is_available() and is_cuda_multiprocessing_enabled() else torch.device("cpu")
        self.device = torch.cuda._get_device(self.device)
        # Set max_new_tokens based on model context window and max_length
        self.max_new_tokens = None  # Will be set after model/tokenizer is loaded

    def init_model(self):
        """Initialize tokenizer and generative model (seq2seq or causal) for finetuning or generation.
        Ensures compatibility with all models and suppresses flash-attention warnings by using attn_implementation="eager" if possible.
        """
        # Avoid reinitialization if already initialized
        if getattr(self, 'model', None) is not None and getattr(self, 'tokenizer', None) is not None:
            return

        model_name = self.inner_model.name
        print(f"Initializing generative model: {model_name}")
        assert isinstance(model_name, str), "Model name must be a string"
        # Load config
        self.config = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=(self.dropout_rate if hasattr(self, "dropout_rate") else 0),
            attention_probs_dropout_prob=(self.dropout_rate if hasattr(self, "dropout_rate") else 0),
            trust_remote_code=True,
        )
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception as e:
            print(f"Error loading tokenizer for model '{model_name}': {e}")
            raise
        # Determine model type: seq2seq or causal LM
        self.is_encoder_decoder = getattr(self.config, 'is_encoder_decoder', False)
        try:
            if self.is_encoder_decoder:
                from transformers import AutoModelForSeq2SeqLM
                self.model = safe_model_from_pretrained(
                    AutoModelForSeq2SeqLM, model_name, config=self.config, trust_remote_code=True
                )
            else:
                from transformers import AutoModelForCausalLM
                self.model = safe_model_from_pretrained(
                    AutoModelForCausalLM, model_name, config=self.config, trust_remote_code=True
                )
        except Exception as e:
            print(f"Error loading generative model for '{model_name}': {e}")
            raise
        # Move model to device
        self.model.to(self.device)
        # For decoder-only models, ensure left padding
        if not self.is_encoder_decoder:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

        # Robustly determine the model's context window for all HuggingFace generative models
        context_window = None
        # Try all common config attributes
        for attr in [
            "n_positions",
            "max_position_embeddings",
            "seq_length",
            "max_seq_len",
            "max_sequence_length",
            "max_position_ids",
            "max_length",
        ]:
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value > 0:
                context_window = value
                break

        # Try tokenizer.model_max_length if not found
        if context_window is None:
            value = getattr(self.tokenizer, "model_max_length", None)
            # Some tokenizers use 1e30 or higher as 'infinite', which is not valid
            if isinstance(value, int) and value > 0 and value < 1e9:
                context_window = value

        # Try model.config.max_length if not found
        if context_window is None:
            value = getattr(self.model.config, "max_length", None)
            if isinstance(value, int) and value > 0:
                context_window = value

        # Fallback to 1024 and warn
        if context_window is None:
            context_window = 1024
            if self.verbose:
                print("[WARN] Could not determine model context window from config or tokenizer. Using fallback value 1024.")

        # Compute max_new_tokens so that input + output <= context_window
        # If max_length is too large, ensure at least 1 token can be generated
        if self.max_length >= context_window:
            self.max_new_tokens = 1
        else:
            self.max_new_tokens = context_window - self.max_length
        if self.verbose:
            print(f"[INFO] Model context window: {context_window}, max_length: {self.max_length}, computed max_new_tokens: {self.max_new_tokens}")

    def _create_dataset(self, X, y):
        return Text2TextDataset(X, y, self.tokenizer, self.max_length)

    def _setup_optimizer(self):
        if self.optimizer_name == "adamw":
            return AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adagrad":
            return torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _setup_scheduler(self, optimizer, total_steps):
        if self.lr_scheduler == "linear":
            return get_linear_schedule_with_warmup(optimizer, self.warmup_steps, total_steps)
        elif self.lr_scheduler == "cosine":
            return get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, total_steps)
        elif self.lr_scheduler == "cosine_with_restarts":
            return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, self.warmup_steps, total_steps)
        elif self.lr_scheduler == "polynomial":
            return get_polynomial_decay_schedule_with_warmup(optimizer, self.warmup_steps, total_steps)
        elif self.lr_scheduler == "constant":
            return get_constant_schedule_with_warmup(optimizer, self.warmup_steps)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler}")

    def _get_num_workers(self):
        if self.num_workers == "default":
            return 0
        elif self.num_workers == "3/4":
            return max(1, int(0.75 * os.cpu_count()))
        elif self.num_workers == "half":
            return max(1, int(0.5 * os.cpu_count()))
        elif self.num_workers == "1/4":
            return max(1, int(0.25 * os.cpu_count()))
        else:
            return 0

    def finetune(self, X, y):
        self.init_model()
        self.print_trainable_parameters()
        dataset = self._create_dataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._get_num_workers(),
        )
        optimizer = self._setup_optimizer()
        total_steps = len(dataloader) * self.epochs
        scheduler = self._setup_scheduler(optimizer, total_steps)
        scaler = torch.amp.GradScaler() if self.use_mixed_precision and self.device.type == "cuda" else None
        previous_loss = None
        epochs_no_improve = 0

        # Use tqdm for epochs and batches, with optimal settings
        epoch_iter = range(self.epochs)
        if self.verbose:
            epoch_iter = tqdm(epoch_iter, desc="Epochs", unit="epoch", leave=True, disable=not self.verbose)

        for epoch in epoch_iter:
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            batch_iter = enumerate(dataloader)
            if self.verbose:
                batch_iter = tqdm(batch_iter, total=len(dataloader), desc=f"Batches (Epoch {epoch+1})", unit="batch", leave=False, disable=not self.verbose)

            for step, batch in batch_iter:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if 'labels' in batch:
                    batch['labels'] = batch['labels'].clone().detach().to(self.device)
                if self.use_mixed_precision and scaler is not None:
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                loss = loss / self.gradient_accumulation_steps
                if self.use_mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                    if self.use_gradient_clipping:
                        if self.use_mixed_precision and scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_max_norm)
                    if self.use_mixed_precision and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                total_loss += loss.detach().item() * self.gradient_accumulation_steps

            avg_loss = total_loss / len(dataloader)
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {avg_loss}")
            # Early stopping
            if previous_loss is not None:
                if previous_loss - avg_loss < self.early_stopping_delta:
                    epochs_no_improve += 1
                    if self.verbose:
                        print(f"No improvement in loss. ({epochs_no_improve}/{self.early_stopping_patience})")
                    if epochs_no_improve >= self.early_stopping_patience:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break
                else:
                    epochs_no_improve = 0
            previous_loss = avg_loss
        return y

    def predict(self, X):
        """Generate texts for input prompts X using the loaded generative model."""
        # Prepare dataset and dataloader
        # Use simple dataset for tokenizing prompts without targets
        dataset = SimpleTextDataset(X, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self._get_num_workers(),
        )
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in dataloader:
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                input_ids = inputs.get('input_ids')
                # Defensive check is no longer needed: max_new_tokens is always computed to fit the context window
                input_length = input_ids.shape[1] if input_ids is not None else 0
                # Generate output ids
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                # Decode generated tokens
                decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                results.extend(decoded)
        return results

    def run(self, X: Seq[Prompt], y: Supervised[Seq[GeneratedText]]) -> Seq[GeneratedText]:
        if self._mode == "train":
            # Perform finetuning
            return self.finetune(X, y)
        else:
            # Ensure model is initialized before prediction
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be initialized and tuned before prediction.")
            # Perform prediction
            return self.predict(X)

    def train(self):
        """Set mode to training for finetuning."""
        self._mode = "train"

    def eval(self):
        """Set mode to evaluation for generation."""
        self._mode = "eval"

@nice_repr
class PartialFineTuneGenLLMTask(FineTunerGenBase):
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        num_trainable_layers: CategoricalValue(1, 2),#, 4, 8, 16, 32, 64),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64),#, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
        verbose: BooleanValue() = True,  # type: ignore
    ):
        super().__init__(
            inner_model,
            batch_size,
            max_length,
            learning_rate,
            epochs,
            warmup_steps,
            weight_decay,
            optimizer,
            gradient_accumulation_steps,
            lr_scheduler,
            use_mixed_precision,
            use_gradient_clipping,
            gradient_clipping_max_norm,
            early_stopping_delta,
            early_stopping_patience,
            num_workers,
            verbose,
        )
        self.num_trainable_layers = num_trainable_layers

    def finetune(self, X, y):
        self.init_model()
        self._set_partial_trainable_layers()
        self.print_trainable_parameters()
        return super().finetune(X, y)

    def _set_partial_trainable_layers(self):
        """
        Freeze all layers, then unfreeze the last N transformer blocks (encoder/decoder) and always unfreeze output heads.
        This works for both encoder-decoder and decoder-only models.
        """
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Comprehensive block/layer patterns for many architectures
        import re
        block_patterns = [
            re.compile(r"\.encoder\.block\.(\d+)\."),      # T5 encoder
            re.compile(r"\.decoder\.block\.(\d+)\."),      # T5 decoder
            re.compile(r"(?:^|\.)h\.(\d+)\."),             # GPT2/Llama/Mistral/Falcon (any prefix or start)
            re.compile(r"\.layer\.(\d+)\."),               # BERT/Roberta
            re.compile(r"\.transformer\.layers\.(\d+)\."), # Some models
            re.compile(r"\.model\.layers\.(\d+)\."),       # Phi, DeepSeek, etc.
            re.compile(r"\.transformer\.blocks\.(\d+)\."), # Qwen, Yi, etc.
            re.compile(r"\.layers\.(\d+)\."),              # Some generic
        ]
        block_indices = set()
        block_name_map = dict()  # idx -> list of names
        for name, _ in self.model.named_parameters():
            for pattern in block_patterns:
                match = pattern.search(name)
                if match:
                    idx = int(match.group(1))
                    block_indices.add(idx)
                    block_name_map.setdefault(idx, []).append(name)
        block_indices = sorted(block_indices, reverse=True)

        # Unfreeze last N blocks
        for idx in block_indices[:self.num_trainable_layers]:
            for name in block_name_map[idx]:
                param = dict(self.model.named_parameters())[name]
                param.requires_grad = True

        # Always unfreeze output heads (lm_head, classifier, final_logits_bias, etc.)
        output_keywords = ["lm_head", "classifier", "final_logits_bias", "score", "output"]
        for name, param in self.model.named_parameters():
            if any(k in name for k in output_keywords):
                param.requires_grad = True

        # Print trainable vs total parameters for transparency
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

@nice_repr
class LoraGenLLMTask(FineTunerGenBase):
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        lora_r: DiscreteValue(1, 32),  # type: ignore
        lora_alpha: CategoricalValue(8, 16, 32, 64),  # type: ignore
        lora_dropout: CategoricalValue(0.0, 0.1, 0.2, 0.3),  # type: ignore
        lora_bias: CategoricalValue("none", "all", "lora_only"),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
        verbose: BooleanValue() = True,  # type: ignore
    ):
        super().__init__(
            inner_model,
            batch_size,
            max_length,
            learning_rate,
            epochs,
            warmup_steps,
            weight_decay,
            optimizer,
            gradient_accumulation_steps,
            lr_scheduler,
            use_mixed_precision,
            use_gradient_clipping,
            gradient_clipping_max_norm,
            early_stopping_delta,
            early_stopping_patience,
            num_workers,
            verbose,
        )
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias

    def init_model(self):
        super().init_model()
        # Automatically determine target_modules for LoRA
        import torch.nn as nn
        import transformers
        # Common module names for attention layers in HF models
        candidate_keywords = [
            "q_proj", "v_proj", "k_proj", "o_proj",  # Llama, Mistral, Falcon, etc.
            "query", "key", "value", "dense",        # BERT, GPT2, T5, etc.
            "attn", "proj", "fc", "mlp"
        ]
        # Collect all unique module names containing these keywords
        target_modules = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                for kw in candidate_keywords:
                    if kw in name:
                        # Only add the last part of the module name (actual layer)
                        last = name.split(".")[-1]
                        target_modules.add(last)
        # Fallback: if nothing found, use all Linear layers
        if not target_modules:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    last = name.split(".")[-1]
                    target_modules.add(last)
        # Convert to sorted list for reproducibility
        target_modules = sorted(target_modules)
        # Apply LoRA using PEFT
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            task_type=TaskType.CAUSAL_LM if not self.is_encoder_decoder else TaskType.SEQ_2_SEQ_LM,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

@nice_repr
class FineTuneGenLLMTask(FineTunerGenBase):
    """
    Modular wrapper for supervised finetuning of generative models
    (encoder-decoder and decoder-only) for text-to-text tasks
    (e.g., summarization, translation, text generation).
    """
    def __init__(
        self,
        inner_model: algorithm(*[Prompt, GeneratedText], include=["transformer"]),  # type: ignore
        batch_size: CategoricalValue(2, 4, 8, 16, 32, 64, 128, 256),  # type: ignore
        max_length: CategoricalValue(64, 128, 256, 512, 1024, 2048, 4096),  # type: ignore
        learning_rate: CategoricalValue(5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4),  # type: ignore
        epochs: DiscreteValue(1, 10),  # type: ignore
        warmup_steps: CategoricalValue(0, 100, 500, 1000, 1500, 2000),  # type: ignore
        weight_decay: CategoricalValue(0, 0.001, 0.005, 0.01, 0.1),  # type: ignore
        optimizer: CategoricalValue("adamw", "adam", "sgd", "adagrad"),  # type: ignore
        gradient_accumulation_steps: CategoricalValue(1, 2, 4, 8, 16),  # type: ignore
        lr_scheduler: CategoricalValue("linear", "cosine", "cosine_with_restarts", "polynomial", "constant"),  # type: ignore
        use_mixed_precision: BooleanValue(),  # type: ignore
        use_gradient_clipping: BooleanValue(),  # type: ignore
        gradient_clipping_max_norm: CategoricalValue(0.5, 1.0, 5.0),  # type: ignore
        early_stopping_delta: CategoricalValue(0.001, 0.005, 0.01),  # type: ignore
        early_stopping_patience: DiscreteValue(1, 10),  # type: ignore
        num_workers: CategoricalValue("default","3/4"),  # type: ignore
        verbose: BooleanValue() = True,  # type: ignore
    ):
        super().__init__(
            inner_model,
            batch_size,
            max_length,
            learning_rate,
            epochs,
            warmup_steps,
            weight_decay,
            optimizer,
            gradient_accumulation_steps,
            lr_scheduler,
            use_mixed_precision,
            use_gradient_clipping,
            gradient_clipping_max_norm,
            early_stopping_delta,
            early_stopping_patience,
            num_workers,
            verbose,
        )