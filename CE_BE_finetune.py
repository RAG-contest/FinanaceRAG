import torch
from torch.utils.data import DataLoader

from sentence_transformers import CrossEncoder, SentenceTransformer, InputExample, losses
from datasets import load_dataset

import pandas as pd
import random
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from tqdm import tqdm
import pickle
import os

from CustomDataset import FinanceDataset
from transformersCL import CrossEncoderCL, SentenceTransformerCL

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
    print(len(total_dataset))
    random.shuffle(total_dataset)

    def in_batch_collate_fn(batch):
        new_batch = [InputExample(texts=[batch[i][0], batch[j][1]], label=float(batch[j][3] not in get_negative_corpus(batch[i][2]))) for i in range(len(batch)) for j in range(len(batch))]
        return new_batch
    
    def in_batch_collate_fn_e5(batch):
        new_batch = [InputExample(texts=["query: "+batch[i][0], "passage: "+batch[j][1]], label=float(batch[j][3] not in get_negative_corpus(batch[i][2]))) for i in range(len(batch)) for j in range(len(batch))]
        return new_batch

    # Split into train and test sets
    num_data = len(total_dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * num_data)

    train_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text)
    test_dataset = FinanceDataset(total_dataset[:train_size], all_corpus, get_id_text, 'test')

    # Define the model
    ce_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'  # Or any suitable model
    bi_model_name = 'intfloat/e5-large-v2'
    ce_model = CrossEncoderCL(ce_model_name)

    print("started fitting")
    # Define DataLoader and loss function & Train Model
    ce_train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10, num_workers=0, pin_memory=True, collate_fn=in_batch_collate_fn)
    ce_train_loss = None#losses.CosineSimilarityLoss(model=model.model)

    ce_model.fit(train_dataloader=ce_train_dataloader, epochs=epoch, loss_fct=ce_train_loss, show_progress_bar=True)
    torch.cuda.empty_cache()

    bi_model = SentenceTransformerCL(bi_model_name)
    bi_train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=17, num_workers=0, pin_memory=True, collate_fn=in_batch_collate_fn_e5)
    bi_train_loss = losses.CosineSimilarityLoss(model=bi_model)

    bi_model.fit(train_objectives=[(bi_train_dataloader, bi_train_loss)], epochs=epoch, show_progress_bar=True)

    ce_model_name = f'{dataset_name}/finance_cross_encoder_model_e{epoch}_b10'
    bi_model_name = f'{dataset_name}/finance_bi_encoder_e{epoch}_b17'
    ce_model.save(f'outputs/models/{ce_model_name}')
    bi_model.save(f'outputs/models/{bi_model_name}')
    ce_model = CrossEncoderCL(f'outputs/models/{ce_model_name}')
    
    # Evaluate on the test set
    print("processing test_data in batch")
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=10, num_workers=0, pin_memory=True, collate_fn=in_batch_collate_fn)
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
    ce_scores = ce_model.predict(list(zip(test_queries, test_corpus)), show_progress_bar=True)
    # bi_scores = bi_model.predict(list(zip(test_queries, test_corpus)), show_progress_bar=True)

    # Compute evaluation metrics
    mse = mean_squared_error(test_labels, ce_scores)
    predicted_labels = [1.0 if score >= 0.5 else 0.0 for score in ce_scores]
    accuracy = accuracy_score(test_labels, predicted_labels)

    print(f'Test MSE: {mse}')
    print(f'Test Accuracy: {accuracy}')
    del test_dataset, train_dataset, ce_model, bi_model

def main():
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
    epoch_num = [10]*4 + [20]*3
    for name, epoch in zip(task_names, epoch_num):
        torch.cuda.empty_cache()
        finetune(name, epoch)

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
    