import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from datasets import load_dataset

import pandas as pd
import random
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle
import os

from CustomDataset import FinanceDataset
from transformersCL import CrossEncoderCL

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

    neg_num = 100
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
        # i for parity bit
        new_batch = [(InputExample(texts=[batch[i][0], batch[j][k][1]], label=int(i==j)),i*len(batch)+j) for i in range(len(batch)) for j in range(len(batch))for k in range(4)]
        return new_batch

    # Define the model
    model_name = 'Capreolus/bert-base-msmarco'  # Or any suitable model
    model = CrossEncoderCL(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Split into train and test sets
    num_data = len(total_dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * num_data)

    train_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text, 'train', tokenizer)
    test_dataset = FinanceDataset(total_dataset[train_size:], all_corpus, get_id_text, 'test')


    # Define DataLoader and loss function
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=0, pin_memory=True, collate_fn=in_batch_collate_fn)
    train_loss = None#losses.CosineSimilarityLoss(model=model.model)

    # Create validation evaluator based on test dataset (flawed since it uses test as validation)
    queries = [q for q, _, _, _ in test_dataset]
    corpus = [c for _, c, _, _ in test_dataset]
    sentence_pair = [(q, c) for q, c, _, _ in test_dataset]
    scores = [1]*len(queries)
    evaluator = CEBinaryClassificationEvaluator(sentence_pair, scores) # for cross encoder. scores == labels
    # evaluator = EmbeddingSimilarityEvaluator(queries, corpus, scores, "cosine") # for bi encoder

    # Train the model
    print("started fitting")
    print(len(train_dataloader))
    model.fit(train_dataloader=train_dataloader, epochs=epoch, loss_fct=train_loss, show_progress_bar=True, evaluator=evaluator, evaluation_steps=len(train_dataloader))

    model_name = 'bert_maxp'
    model.save(f'outputs/models/{dataset_name}/{model_name}')
    model = CrossEncoderCL(f'outputs/models/{dataset_name}/{model_name}')
        
    # 검증 절차
    print("Starting validation...")
    test_queries = [q for q, c in test_dataset]
    test_corpus = [c for q, c in test_dataset]
    true_labels = [1.0] * len(test_queries)  # 긍정 샘플만 있다고 가정
    
    predicted_scores = []
    for query, corpus in tqdm(zip(test_queries, test_corpus), total=len(test_queries)):
        score = model.predict([(query, corpus)])
        predicted_scores.append(score[1])
    
    # 성능 지표 계산
    acc = accuracy_score([1 if s > 0.5 else 0 for s in predicted_scores], true_labels)
    roc_auc = roc_auc_score(true_labels, predicted_scores)

    print(f"Validation Results for {dataset_name}:")
    print(f"Accuracy: {acc}")
    print(f"ROC-AUC: {roc_auc}")

def main():
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
    epoch_num = [10]*7
    for name, epoch in zip(task_names, epoch_num):
        torch.cuda.empty_cache()
        finetune(name, epoch)

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()