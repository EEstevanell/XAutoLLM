import json
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
import re
from enum import Enum
from tqdm import tqdm
import requests
from torch.utils.data import Dataset, DataLoader
import torch
import os
    
class TASK_ALIASES(Enum):
    TextClassification = "text-classification"
    TokenClassification = "token-classification"
    WordEmbeddings = "word-embeddings"
    TextGeneration = "text-generation"

class DOWNLOAD_MODE(Enum):
    HUB = "hub"
    BASE = "base"
    SCRAP = "scrap"
    
class ModelDescriptor():
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
    ]
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
                    "id2label": config.id2label if hasattr(config, "id2label") else None,
                    "model_type": config.model_type if hasattr(config, "model_type") else None,
                    "architectures": config.architectures if hasattr(config, "architectures") else None,
                    "vocab_size": config.vocab_size if hasattr(config, "vocab_size") else None,
                    "type_vocab_size": config.type_vocab_size if hasattr(config, "type_vocab_size") else None,
                    "is_decoder": config.is_decoder if hasattr(config, "is_decoder") else None,
                    "is_encoder_decoder": config.is_encoder_decoder if hasattr(config, "is_encoder_decoder") else None,
                    "num_layers": config.num_hidden_layers if hasattr(config, "num_hidden_layers") else None,
                    "hidden_size": config.hidden_size if hasattr(config, "hidden_size") else None,
                    "num_attention_heads": config.num_attention_heads if hasattr(config, "num_attention_heads") else None,
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
    units = {'k': 1000, 'M': 1000000}
    if s[-1] in units:
        return float(s[:-1]) * units[s[-1]]
    else:
        return float(s)

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
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        encoding = {key: val.squeeze() for key, val in encoding.items()}

        if self.labels is not None:
            label = self.labels[idx]
            encoding['labels'] = torch.tensor(label, dtype=torch.long)

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
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                target_enc = self.tokenizer(
                    self.targets[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt',
                )
                item = {key: val.squeeze() for key, val in input_enc.items()}
                item['labels'] = target_enc['input_ids'].squeeze()
                return item