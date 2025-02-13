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
        
        # gemma
        # "google/gemma-7b-it",
        # "google/gemma-2b-it",
        # "google/gemma-7b",
        # "google/gemma-2b",
        
        # GPT-2
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        
        # BART
        "facebook/bart-base",
        "facebook/bart-large",
        
        # PHI
        "microsoft/Phi-3-small-8k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/phi-2",
        "microsoft/phi-1_5",
        
        # Mistral
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1"
    ]
}

def get_base_hf_models(target_task):
    if target_task in TASK_TO_BASE_MODELS:
        return TASK_TO_BASE_MODELS[target_task]
    else:
        return []
    
def get_hf_models(target_task):
    hf_api = HfApi()
    return hf_api.list_models(task=target_task, library="pytorch")

def get_hf_models_sorted_by_likes(target_task, min_likes, min_downloads):
    from bs4 import BeautifulSoup
    page = 0
    count = 0
    
    while True:
        url = f"https://huggingface.co/models?pipeline_tag={target_task}&sort=likes"
        if page > 0:
            url += f"&p={page}"

        response = requests.get(url)
        
        soup = BeautifulSoup(response.content.decode('utf8'))

        for model in soup.find_all('article'):

            parsed_text = [line.strip() for line in re.sub(' +', ' ', model.text.replace('\n', ' ').replace('\t', ' ').replace('•', '\n')).strip().split('\n')]
            model_name_str, last_updated_str, downloaded, *likes = parsed_text
            likes = int(likes[0]) if likes else 0
            downloads = convert_string_to_number(downloaded.strip())
            
            if (downloads < min_downloads):
                return
            
            if (likes < min_likes):
                return

            model_name = model.find('a').attrs['href'][1:]
            timestamp = model.find('time').attrs['datetime']
            yield ModelDescriptor(model_name, downloads, likes, target_task)

            count += 1
        page += 1

def get_model_config(modelId):
    config = AutoConfig.from_pretrained(modelId, use_auth_token=os.getenv('HUGGINGFACE_HUB_TOKEN'), trust_remote_code=True)
    return config

def get_models_info(target_task, max_amount, min_likes=None, min_downloads=None, download_mode=DOWNLOAD_MODE.HUB):
    models = get_hf_models(target_task.value) \
        if download_mode == DOWNLOAD_MODE.HUB \
        else get_base_hf_models(target_task) if DOWNLOAD_MODE.BASE \
        else get_hf_models_sorted_by_likes(target_task.value, min_likes, min_downloads)
    
    # regex for detecting partially trained models
    pattern = r"train-\d+"
    
    # Get model metadata
    model_info = []
    current = 0
    for model in tqdm(models):
        if current >= max_amount:
            break
        
        modelId = model if download_mode == DOWNLOAD_MODE.BASE else model.modelId
        if re.search(pattern, modelId) is not None:
            continue

        try:
            if download_mode == DOWNLOAD_MODE.SCRAP:
                likes, downloads = model.likes, model.downloads
            else:
                if download_mode == DOWNLOAD_MODE.HUB:
                    likes, downloads = get_model_likes_downloads(model.modelId)
            
            if (download_mode == DOWNLOAD_MODE.SCRAP and min_likes is not None and likes < min_likes):
                continue 
            
            if (download_mode == DOWNLOAD_MODE.SCRAP and min_downloads is not None and downloads < min_downloads):
                continue 
            
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
            
            if download_mode == DOWNLOAD_MODE.SCRAP:
                info["metadata"]["likes"] = likes
                info["metadata"]["downloads"] = downloads
                
            model_info.append(info)
            current += 1
        except Exception as e:
           print(e)
    return model_info

def download_models_info(
    target_task, 
    max_amount=1000, 
    min_likes=100, 
    min_downloads=1000,
    download_mode=DOWNLOAD_MODE.HUB
):
    # Get model info and dump to JSON file
    model_info = get_models_info(target_task, max_amount, min_likes, min_downloads, download_mode=download_mode)
    with open(f"{target_task.value}.json", "w") as f:
        json.dump(model_info, f)
        print(f"Model information has been saved to {target_task.value}.json")

    return model_info

def get_model_likes_downloads(model_name):
    url = f"https://huggingface.co/{model_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the HTML element with the likes
    likes_element = soup.find('button', {'title': 'See users who liked this repository'})
    likes = int(likes_element.text) if likes_element else 0

    # Find the HTML element with the downloads
    downloads_element = soup.find('dt', text='Downloads last month').find_next_sibling('dd')
    downloads = int(downloads_element.text.replace(',', '')) if downloads_element else 0

    return likes, downloads

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