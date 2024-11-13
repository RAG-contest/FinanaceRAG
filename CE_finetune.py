import torch
from torch.utils.data import DataLoader

from sentence_transformers import CrossEncoder, InputExample, losses
from datasets import load_dataset

import pandas as pd
import random
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle
import os

from CustomDataset import FinanceDataset
from CrossEncoderCL import CrossEncoderCL

def main():
    # Load the dataset
    ds = load_dataset("Linq-AI-Research/FinanceRAG", "MultiHiertt")

    # Load ground truth relevance judgments
    df = pd.read_csv('gt/MultiHiertt_qrels.tsv', sep='\t')

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
    dataset_path = "total_dataset_positive_pair"
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

        # print("mapping id to corpus..")
        # total_dataset = []
        # for example in tqdm(dataset_examples):
        #     q_id = example['query_id']
        #     c_id = example['corpus_id']
        #     label = example['label']
        #     query_text = get_id_text('queries', q_id)
        #     corpus_text = get_id_text('corpus', c_id)
        #     if query_text is None or corpus_text is None:
        #         continue  # Skip if text not found
        #     input_example = InputExample(texts=[query_text, corpus_text], label=label)
        #     total_dataset.extend([input_example])
        total_dataset = [(di['query_id'], di['corpus_id']) for di in dataset_examples]

        print(f"saving {dataset_path} as pickle..")
        with open(f'preprocessed/{dataset_path}.pkl', 'wb') as f:
            pickle.dump(total_dataset, f)
        print("Done.\n")

    random.shuffle(total_dataset)

    def in_batch_collate_fn(batch):
        new_batch = [InputExample(texts=[batch[i][0], batch[j][1]], label=float(batch[j][3] not in get_negative_corpus(batch[i][2]))) for i in range(len(batch)) for j in range(len(batch))]
        return new_batch

    # Split into train and test sets
    num_data = len(total_dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * num_data)

    train_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text)
    test_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text, 'test')

    # Define the model
    model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'  # Or any suitable model
    model = CrossEncoderCL(model_name)

    # Define DataLoader and loss function
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=12, num_workers=0, pin_memory=True, collate_fn=in_batch_collate_fn)
    train_loss = None#losses.CosineSimilarityLoss(model=model.model)

    # Train the model
    print("started fitting")
    # model.fit(train_dataloader=train_dataloader, epochs=5, loss_fct=train_loss, show_progress_bar=True)

    model_name = 'finance_cross_encoder_model_e5_b10'
    # model.save(f'outputs/models/{model_name}')
    model = CrossEncoderCL(f'outputs/models/{model_name}')
        
    # Evaluate on the test set
    print("processing test_data in batch")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=12, num_workers=0, pin_memory=True, collate_fn=in_batch_collate_fn)
    temp = [batch for batch in tqdm(test_loader)]
    batchs = []
    for batch in temp:
        batchs += batch
    
    print("unpacking")
    test_queries = [example.texts[0] for example in tqdm(batchs)]
    test_corpus = [example.texts[1] for example in tqdm(batchs)]
    test_labels = [example.label for example in tqdm(batchs)]

    # Compute predictions
    print("predicting model score")
    scores = model.predict(list(zip(test_queries, test_corpus)), show_progress_bar=True)

    # Compute evaluation metrics
    mse = mean_squared_error(test_labels, scores)
    predicted_labels = [1.0 if score >= 0.5 else 0.0 for score in scores]
    accuracy = accuracy_score(test_labels, predicted_labels)

    print(f'Test MSE: {mse}')
    print(f'Test Accuracy: {accuracy}')

if __name__ == "__main__":
    main()