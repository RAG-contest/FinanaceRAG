import os
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import losses, InputExample, models
from transformersCL import SentenceTransformerCL
import random
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import pickle  # For saving and loading preprocessed data
import torch.nn as nn

def train(task_names):
    # Path to save/load preprocessed data
    preprocessed_data_path = 'preprocessed/positive_pairs.pkl'

    # Check if preprocessed data exists
    if os.path.exists(preprocessed_data_path):
        print("Loading preprocessed data from", preprocessed_data_path)
        with open(preprocessed_data_path, 'rb') as f:
            positive_pairs = pickle.load(f)
    else:
        print("Preprocessing data and saving to", preprocessed_data_path)
        # Combined positive_pairs list
        positive_pairs = []
        
        for dataset_name in task_names:
            print(f"Processing dataset: {dataset_name}")
            # 1. Load the dataset
            dataset = load_dataset('Linq-AI-Research/FinanceRAG', dataset_name)
    
            # 2. Load the CSV file
            qrels_df = pd.read_csv(f'gt/{dataset_name}_qrels.tsv', sep='\t', header=None, names=['query_id', 'corpus_id', 'score'])
    
            # 3. Map text based on '_id' in corpus and queries datasets
            corpus_df = pd.DataFrame(dataset['corpus'])
            queries_df = pd.DataFrame(dataset['queries'])
    
            query_id_to_text = pd.Series(queries_df.text.values, index=queries_df._id).to_dict()
            corpus_id_to_text = pd.Series(corpus_df.text.values, index=corpus_df._id).to_dict()
    
            # 4. Create a list of positive pairs
            for _, row in qrels_df.iterrows():
                query_id = row['query_id']
                corpus_id = row['corpus_id']
                query_text = query_id_to_text.get(query_id, "")
                corpus_text = corpus_id_to_text.get(corpus_id, "")
                if query_text and corpus_text:
                    positive_pairs.append((query_text, corpus_text))
        
        print(f"Total positive pairs: {len(positive_pairs)}")
        
        # Save preprocessed data
        os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
        with open(preprocessed_data_path, 'wb') as f:
            pickle.dump(positive_pairs, f)
    
    # 5. Define PyTorch Dataset class
    class FinanceRAGDataset(Dataset):
        def __init__(self, positive_pairs):
            self.positive_pairs = positive_pairs
    
        def __len__(self):
            return len(self.positive_pairs)
    
        def __getitem__(self, idx):
            query, positive = self.positive_pairs[idx]
            return {'query': query, 'positive': positive}
    
    # 6. Define custom collate_fn
    def in_batch_collate_fn(batch):
        """
        Use all positive pairs in the batch as negative pairs within the batch
        """
        input_examples = []
        batch_size = len(batch)
        for i in range(batch_size):
            query = batch[i]['query']
            positive = batch[i]['positive']
            # Add positive sample
            input_examples.append(InputExample(texts=[query, positive], label=1.0))
            for j in range(batch_size):
                if i != j:
                    negative = batch[j]['positive']
                    # Add negative sample
                    input_examples.append(InputExample(texts=[query, negative], label=0.0))
        return input_examples
    
    # 7. Load the model
    model = SentenceTransformerCL('dunzhang/stella_en_400M_v5', trust_remote_code=True, prompts={"query": "s2p_query", "passage": ""})
    
    # 8. Create Dataset and DataLoader
    train_dataset = FinanceRAGDataset(positive_pairs)
    batch_size = 16  # Adjust as needed
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=in_batch_collate_fn)
    
    # 9. Define the loss function
    train_loss = losses.ContrastiveLoss(model=model)
    
    # 10. Fine-tuning settings
    num_epochs = 20
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    
    # 11. Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=f'.models/{dataset_name}/stella_en_e20'
    )
    
    # 12. Save the model
    model.save(f'.models/{dataset_name}/stella_en_e20')
    
if __name__ == "__main__":
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
    train(task_names)
