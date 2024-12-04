import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder, InputExample, losses
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from datasets import load_dataset

import pandas as pd
import random
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle

from CustomDataset import FinanceDataset
from transformersCL import CrossEncoderCL

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    print("cleaned")

def finetune(rank, world_size, dataset_name, epoch):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Load the dataset
    ds = load_dataset("Linq-AI-Research/FinanceRAG", dataset_name)

    # Load ground truth relevance judgments
    df = pd.read_csv(f'gt/{dataset_name}_qrels.tsv', sep='\t')

    # Get all unique query IDs and corpus IDs
    all_queries = df['query_id'].unique()
    all_corpus = ds['corpus']['_id']

    if rank == 0:
        print("Generating ID lookup...")
    # Create mappings from IDs to indices for fast lookup
    query_id_to_index = {id_: idx for idx, id_ in enumerate(ds['queries']['_id'])}
    corpus_id_to_index = {id_: idx for idx, id_ in enumerate(ds['corpus']['_id'])}
    if rank == 0:
        print("Done.\n")

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
        if rank == 0:
            print("Dataset mapping exists!")
            print("Loading total_dataset from pickle...\n")
        with open(f'preprocessed/{dataset_path}.pkl', 'rb') as f:
            total_dataset = pickle.load(f)
    else:
        if rank == 0:
            print("Preprocessed data doesn't exist")
            print("Generating positive pairs...")
        dataset_examples = []
        for query_id in tqdm(all_queries, disable=rank != 0):
            positive_corpus_ids = df[(df['query_id'] == query_id) & (df['score'] == 1)]['corpus_id'].tolist()
            for corpus_id in positive_corpus_ids:
                dataset_examples.append({'query_id': query_id, 'corpus_id': corpus_id, 'label': 1.0})
        if rank == 0:
            print("Done.\n")
        total_dataset = [(di['query_id'], di['corpus_id']) for di in dataset_examples]

        if rank == 0:
            print(f"Saving {dataset_path} as pickle...")
            with open(f'preprocessed/{dataset_path}.pkl', 'wb') as f:
                pickle.dump(total_dataset, f)
            print("Done.\n")

    random.shuffle(total_dataset)

    def in_batch_collate_fn(batch):
        # i for parity bit
        new_batch = [(InputExample(texts=[batch[i][0], batch[j][k][1]], label=int(i==j)),i*len(batch)+j) for i in range(len(batch)) for j in range(len(batch))for k in range(4)]
        return new_batch

    # Define the model
    model_name = 'Capreolus/bert-base-msmarco'  # Or any suitable model
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    pipeline = CrossEncoderCL(model_name)
    model = pipeline.model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    pipeline.model = model  # Update the model in the pipeline

    # Split into train and test sets
    num_data = len(total_dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * num_data)

    train_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text, 'train', tokenizer)
    test_dataset = FinanceDataset(total_dataset[train_size:], all_corpus, get_id_text, 'test', tokenizer)

    # Define DistributedSampler for training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True,
        collate_fn=in_batch_collate_fn,
        sampler=train_sampler
    )

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Create evaluator on rank 0
    evaluator = None
    if rank == 0:
        queries = [q for q, _, _, _ in test_dataset]
        corpus = [c for _, c, _, _ in test_dataset]
        sentence_pair = [(q, c) for q, c in zip(queries, corpus)]
        scores = [1] * len(queries)
        evaluator = CEBinaryClassificationEvaluator(sentence_pair, scores)

    # Train the model
    if rank == 0:
        print("Started fitting")
        print(f"Number of training steps: {len(train_dataloader) * epoch}")

    pipeline.fit(
        train_dataloader=train_dataloader,
        epochs=epoch,
        show_progress_bar=rank == 0
    )

    # Save the model
    if rank == 0:
        model_name = 'bert_maxp'
        path = f'models/{dataset_name}/{model_name}'
        model.module.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print("Model saved.")

    cleanup()

def main():
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"][1:]
    epoch_num = [15] * len(task_names)

    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    mp.spawn(
        run_finetune,
        args=(world_size, task_names, epoch_num),
        nprocs=world_size,
        join=True
    )

def run_finetune(rank, world_size, task_names, epoch_num):
    import time
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)
    for name, epoch in zip(task_names, epoch_num):
        torch.cuda.empty_cache()
        print(name) if rank == 0 else None
        finetune(rank, world_size, name, epoch)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    os.environ["WANDB_DISABLED"] = "true"
    main()
