import json
import warnings
from transformers import AutoConfig
import re
from enum import Enum
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import logging  # For logging warnings

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TASK_ALIASES(Enum):
    TextClassification = "text-classification"
    TokenClassification = "token-classification"
    WordEmbeddings = "word-embeddings"
    TextGeneration = "text-generation"


class DOWNLOAD_MODE(Enum):
    HUB = "hub"
    BASE = "base"
    SCRAP = "scrap"


class ModelDescriptor:
    def __init__(self, modelId, downloads, likes, pipeline_tag):
        self.modelId = modelId
        self.downloads = downloads
        self.likes = likes
        self.pipeline_tag = pipeline_tag


TASK_TO_BASE_MODELS = {
    TASK_ALIASES.WordEmbeddings: [
        # BERT
        "bert-base-uncased",
        "bert-base-cased",
        "bert-large-uncased",
        "bert-large-cased",
        "bert-base-multilingual-uncased",
        "bert-base-multilingual-cased",
        # RoBERTuito
        "pysentimiento/robertuito-base-uncased",
        "PlanTL-GOB-ES/roberta-base-bne",
        # DistilBERT
        "distilbert-base-uncased",
        "distilbert-base-cased",
        "distilbert-base-multilingual-cased",
        # RoBERTa
        "roberta-base",
        "roberta-large",
        # Deberta
        "microsoft/deberta-v3-base",
        "microsoft/deberta-base",
        "microsoft/mdeberta-v3-base",
        # ALBERT
        "albert-base-v1",
        "albert-large-v1",
        "albert-xlarge-v1",
        "albert-xxlarge-v1",
        # ELECTRA
        "google/electra-small-discriminator",
        "google/electra-base-discriminator",
        "google/electra-large-discriminator",
        # XLM-RoBERTa
        "xlm-roberta-base",
        "xlm-roberta-large",
    ],
    TASK_ALIASES.TextGeneration: [
        ## ENCODER-DECODER
        # t5
        "google-t5/t5-small",
        "google-t5/t5-base",
        "google-t5/t5-large",
        "google-t5/t5-3b",
        "google-t5/t5-11b",
        # flan-t5
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xxl",
        "google/flan-t5-xl",
        # BART
        "facebook/bart-base",
        "facebook/bart-large",
        ## DECODER-ONLY
        # gemma
        "google/gemma-3-4b-it",
        "google/gemma-3-4b-pt",
        "google/gemma-3-12b-it",
        "google/gemma-3-12b-pt",
        "google/gemma-3-27b-it",
        "google/gemma-3-27b-pt",
        "google/gemma-3-1b-it",
        "google/gemma-3-1b-pt",
        # deepseek
        "deepseek-ai/DeepSeek-V2-Lite",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # GPT-2
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        # PHI
        "microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "microsoft/Phi-4",
        "microsoft/Phi-4-reasoning",
        "microsoft/Phi-4-mini-reasoning",
        "microsoft/Phi-3-medium-4k-instruct",
        # Mistral
        "mistralai/Mistral-Nemo-Instruct-FP8-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Base-2407",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1",
        # LLAMA
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
    ],
}


def get_base_hf_models(target_task):
    if target_task in TASK_TO_BASE_MODELS:
        return TASK_TO_BASE_MODELS[target_task]
    else:
        return []


def get_model_config(modelId):
    config = AutoConfig.from_pretrained(modelId, trust_remote_code=True)
    return config


def get_models_info(target_task):
    models = get_base_hf_models(target_task)

    # regex for detecting partially trained models
    pattern = r"train-\d+"

    # Get model metadata
    model_info = []
    current = 0
    for model in tqdm(models):
        modelId = model
        if re.search(pattern, modelId) is not None:
            continue

        try:
            config = get_model_config(modelId)

            info = {
                "name": modelId,
                "metadata": {
                    "task": target_task.value,
                    "id2label": (
                        config.id2label if hasattr(config, "id2label") else None
                    ),
                    "model_type": (
                        config.model_type if hasattr(config, "model_type") else None
                    ),
                    "architectures": (
                        config.architectures
                        if hasattr(config, "architectures")
                        else None
                    ),
                    "vocab_size": (
                        config.vocab_size if hasattr(config, "vocab_size") else None
                    ),
                    "type_vocab_size": (
                        config.type_vocab_size
                        if hasattr(config, "type_vocab_size")
                        else None
                    ),
                    "is_decoder": (
                        config.is_decoder if hasattr(config, "is_decoder") else None
                    ),
                    "is_encoder_decoder": (
                        config.is_encoder_decoder
                        if hasattr(config, "is_encoder_decoder")
                        else None
                    ),
                    "num_layers": (
                        config.num_hidden_layers
                        if hasattr(config, "num_hidden_layers")
                        else None
                    ),
                    "hidden_size": (
                        config.hidden_size if hasattr(config, "hidden_size") else None
                    ),
                    "num_attention_heads": (
                        config.num_attention_heads
                        if hasattr(config, "num_attention_heads")
                        else None
                    ),
                },
            }

            model_info.append(info)
            current += 1
        except Exception as e:
            print(e)
    return model_info


def download_models_info(
    target_task,
):
    # Get model info and dump to JSON file
    model_info = get_models_info(target_task)
    with open(f"{target_task.value}.json", "w") as f:
        json.dump(model_info, f)
        print(f"Model information has been saved to {target_task.value}.json")

    return model_info


def to_camel_case(name):
    # Remove numbers at the beginning, replace '/' with '_', and split on '-'
    words = re.sub(r"^[0-9]*", "", name.replace("/", "_").replace(".", "")).split("-")
    return "".join(re.sub(r"^[0-9]*", "", word).title() for word in words)


def convert_string_to_number(s):
    """
    Convert a string to a number, where the string can end in 'k' or 'M' to signify thousands or millions.
    """
    units = {"k": 1000, "M": 1000000}
    if s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    else:
        return float(s)


def safe_model_from_pretrained(cls, model_name_or_path, **kwargs):
    """
    Utility to load HuggingFace models with attn_implementation="eager" if supported.
    Falls back gracefully if not supported (for compatibility with all models).
    Also supports BitsAndBytes quantization if quantization_config is provided.
    """
    attn_kwargs = dict(attn_implementation="eager")
    # If quantization_config is present, pass it through
    if "quantization_config" in kwargs and kwargs["quantization_config"] is not None:
        attn_kwargs["quantization_config"] = kwargs.pop("quantization_config")
    try:
        return cls.from_pretrained(
            model_name_or_path, **attn_kwargs, **kwargs
        )
    except TypeError as e:
        # attn_implementation or quantization_config not supported
        return cls.from_pretrained(model_name_or_path, **kwargs)
    except Exception as e:
        warnings.warn(
            f"Could not set attn_implementation='eager' or quantization: {e}. Proceeding without it."
        )
        return cls.from_pretrained(model_name_or_path, **kwargs)


class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze() for key, val in encoding.items()}

        if self.labels is not None:
            label = self.labels[idx]
            encoding["labels"] = torch.tensor(label, dtype=torch.long)

        return encoding


class Text2TextDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze() for key, val in input_enc.items()}
        labels = target_enc["input_ids"].squeeze()
        # Mask pad tokens so they’re ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        item["labels"] = labels
        return item


class DecoderOnlyDataset(Dataset):
    """
    Generic dataset for decoder-only models for generative tasks.
    It tokenizes the full text (prompt + target) and creates labels
    where prompt tokens are masked to ensure loss is computed only on target tokens.

    Args:
        inputs (list[str]): List of input prompts (e.g., "Translate to French: Hello").
        targets (list[str]): List of target completions (e.g., "Bonjour").
        tokenizer: HuggingFace tokenizer for the decoder-only model.
                             It's crucial that tokenizer.pad_token is set. If not,
                             it will be set to tokenizer.eos_token.
        max_length (int, optional): Maximum sequence length for tokenized inputs.
                                    Sequences longer than this will be truncated.
                                    If None, sequences are padded to the longest in the batch by the collator.
        prompt_suffix (str, optional): A suffix to append to the input prompt before the target.
                                       Useful for cueing the model (e.g., " Answer: ").
    """

    def __init__(self, inputs, targets, tokenizer, max_length=None, prompt_suffix=""):
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # print("Warning: tokenizer.pad_token was not set. Using tokenizer.eos_token as pad_token.")

        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_suffix = prompt_suffix

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx in self.processed_items:  # Basic caching, can be more sophisticated
            return self.processed_items[idx]

        prompt_text = self.inputs[idx] + self.prompt_suffix
        target_text = self.targets[idx]

        # If target_text is empty, it's inherently problematic.
        if not target_text.strip():
            logger.warning(
                f"Sample {idx} has an empty or whitespace-only target. Skipping this sample. Prompt: '{prompt_text[:100]}...'"
            )
            self.processed_items[idx] = None  # Cache None
            return None

        full_text = prompt_text + target_text

        tokenized_full_text = self.tokenizer(
            full_text,
            truncation=True if self.max_length is not None else False,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = tokenized_full_text[self.input_ids_field]

        tokenized_prompt = self.tokenizer(
            prompt_text,
            truncation=True if self.max_length is not None else False,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False,
        )
        prompt_token_len = len(tokenized_prompt[self.input_ids_field])

        # If prompt_token_len is 0 for a non-empty prompt_text, tokenizer might be misconfigured
        # or prompt_text is unusual (e.g., only special tokens that get stripped).
        if not prompt_text.strip() and prompt_token_len == 0:  # Empty prompt is fine
            pass
        elif prompt_text.strip() and prompt_token_len == 0:
            logger.warning(
                f"Sample {idx} has a non-empty prompt_text ('{prompt_text[:100]}...') but tokenized_prompt_len is 0. "
                "Check tokenizer or prompt content. Treating as no prompt."
            )

        labels = list(input_ids)
        actual_prompt_mask_len = min(prompt_token_len, len(input_ids))

        for i in range(actual_prompt_mask_len):
            labels[i] = -100  # Mask prompt tokens

        # Check if all labels are -100 (i.e., no target tokens after truncation)
        # This happens if full_text is truncated such that only prompt tokens remain,
        # or if the prompt itself filled up max_length.
        if actual_prompt_mask_len == len(input_ids) and len(input_ids) > 0:
            logger.warning(
                f"Sample {idx} contains only prompt tokens after tokenization/truncation "
                f"(prompt_token_len: {prompt_token_len}, input_ids_len: {len(input_ids)}, max_length: {self.max_length}). "
                f"This sample will be skipped. Prompt: '{prompt_text[:100]}...'"
            )
            self.processed_items[idx] = None  # Cache None
            return None

        # Check if there are any valid labels to learn from
        if (
            all(label == -100 for label in labels) and labels
        ):  # Ensure labels is not empty
            logger.warning(
                f"Sample {idx} has no valid target labels after masking (all labels are -100). "
                f"This sample will be skipped. Prompt: '{prompt_text[:100]}...'"
            )
            self.processed_items[idx] = None  # Cache None
            return None

        item = {
            self.input_ids_field: torch.tensor(input_ids, dtype=torch.long),
            self.attention_mask_field: torch.tensor(
                tokenized_full_text[self.attention_mask_field], dtype=torch.long
            ),
            self.labels_field: torch.tensor(labels, dtype=torch.long),
        }
        self.processed_items[idx] = item  # Cache valid item
        return item
