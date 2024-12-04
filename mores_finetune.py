import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from sentence_transformers import losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from datasets import load_dataset

import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle
import os

from CustomDataset import FinanceDataset
from mores import MORES


def get_lambda_scheduler(optimizer, epochs, train_dataloader, warmup_ratio=0.1):
    """
    Creates a LambdaLR scheduler for warmup followed by linear decay.

    Args:
        optimizer: PyTorch optimizer.
        epochs: Total number of epochs.
        train_dataloader: DataLoader used for training.
        warmup_ratio: Fraction of total steps used for warmup (default: 0.1).

    Returns:
        LambdaLR scheduler.
    """
    total_steps = epochs * len(train_dataloader)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / warmup_steps
        else:
            # Linear decay
            return max(0.0, (total_steps - current_step) / (total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)

def train(model, train_dataloader, epochs, sub_batch_size=8):
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA device not detected")
    
    model = model.to(device)
    activation_fct = nn.Identity()
    criterion = nn.BCEWithLogitsLoss()
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in ["bias", "LayerNorm.bias", "LayerNorm.weight"])],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in ["bias", "LayerNorm.bias", "LayerNorm.weight"])], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    scheduler = get_lambda_scheduler(optimizer, epochs, train_dataloader, 0.1)
    
    batch_size = train_dataloader.batch_size
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_dataloader)
        
        for batch_idx, examples in enumerate(pbar):
            for i in range(0, len(examples), sub_batch_size):
                sub_examples = examples[i:i+sub_batch_size]
                
                queries = [example[0] for example in sub_examples]
                corpus = [example[1] for example in sub_examples]
                labels = torch.tensor([example[2] for example in sub_examples], dtype=torch.float32).to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                scores = activation_fct(model(corpus, queries))
                scores = scores.view(-1)
                
                # Compute loss
                loss = criterion(scores, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item() * len(sub_examples)
                
                # Free memory
                del queries, corpus, labels, scores, loss
                torch.cuda.empty_cache()
                
                pbar.set_description(f"Train: [{epoch + 1}] ")
            del examples
            
        avg_epoch_loss = epoch_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg_epoch_loss:.4f}")
    
    torch.cuda.empty_cache()


def finetune(dataset_name, epoch):
    # Load the dataset
    ds = load_dataset("Linq-AI-Research/FinanceRAG", dataset_name)

    # Load ground truth relevance judgments
    df = pd.read_csv(f'gt/{dataset_name}_qrels.tsv', sep='\t')

    # Get all unique query IDs and corpus IDs
    all_queries = df['query_id'].unique()
    all_corpus = ds['corpus']['_id']

    print("generating id lookup..")
    # Create mappings from IDs to indices for fast lookup
    query_id_to_index = {id_: idx for idx, id_ in enumerate(tqdm(ds['queries']['_id']))}
    corpus_id_to_index = {id_: idx for idx, id_ in enumerate(tqdm(ds['corpus']['_id']))}
    print("Done.\n")

    def get_negative_corpus(query_id):
        positive_corpus_ids = df[(df['query_id'] == query_id) & (df['score'] == 1)]['corpus_id'].tolist()
        negative_corpus = list(set(all_corpus) - set(positive_corpus_ids))
        random.shuffle(negative_corpus)
        return set(negative_corpus)

    def get_id_text(id_type, id_):
        if id_type == 'queries':
            idx = query_id_to_index.get(id_)
            if idx is not None:
                return ds['queries']['text'][idx]
        elif id_type == 'corpus':
            idx = corpus_id_to_index.get(id_)
            if idx is not None:
                return ds['corpus']['text'][idx]
        return None

    # Create InputExamples
    dataset_path = f"{dataset_name}_positive_pair"
    if os.path.exists(f'preprocessed/{dataset_path}.pkl'):
        print("dataset mapping exist!")
        with open(f'preprocessed/{dataset_path}.pkl', 'rb') as f:
            total_dataset = pickle.load(f)
        print("loaded total_dataset from pkl\n")

    else:
        print("preprocessed data doesn't exists")
        print("generationg positive pair...")
        # print("generationg positive & negative pair...")
        dataset_examples = []
        for query_id in tqdm(all_queries):
            positive_corpus_ids = df[(df['query_id'] == query_id) & (df['score'] == 1)]['corpus_id'].tolist()
            # negative_corpus_ids = list(get_negative_corpus(query_id))[:neg_num]
            
            for corpus_id in positive_corpus_ids:
                dataset_examples.append({'query_id': query_id, 'corpus_id': corpus_id, 'label': 1.0})
            
            # for corpus_id in negative_corpus_ids:
            #     dataset_examples.append({'query_id': query_id, 'corpus_id': corpus_id, 'label': 0.0})
        print("Done.\n")

        total_dataset = [(di['query_id'], di['corpus_id']) for di in dataset_examples]

        print(f"saving {dataset_path} as pickle..")
        with open(f'preprocessed/{dataset_path}.pkl', 'wb') as f:
            pickle.dump(total_dataset, f)
        print("Done.\n")

    random.shuffle(total_dataset)

    def in_batch_collate_fn(batch):
        new_batch = [(batch[i][0], batch[j][1], float(i==j)) for i in range(len(batch)) for j in range(len(batch))]#float(batch[j][3] not in get_negative_corpus(batch[i][2]))
        return new_batch

    # Split into train and test sets
    num_data = len(total_dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * num_data)

    train_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text)
    test_dataset = FinanceDataset(total_dataset[train_size:], all_corpus, get_id_text, 'test')

    # Define the model
    model_name = 'facebook/bart-large'  # Or any suitable model
    model = MORES(model_name, 8, 256, 2)

    # Define DataLoader and loss function
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=8,
        collate_fn=in_batch_collate_fn
    )
    train(model, train_dataloader, epoch)
    torch.save(model.state_dict(), f"models/{dataset_name}/mores+_bart_unfreeze")
    
if __name__ == "__main__":
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"][3:]
    epochs = [7]*4
    for task_name, epoch in zip(task_names, epochs):
        print(task_name)
        finetune(task_name, epoch)