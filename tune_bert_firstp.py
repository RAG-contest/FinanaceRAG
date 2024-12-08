import os
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import losses, InputExample, models, CrossEncoder
from transformersCL import SentenceTransformerCL
import random
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import pickle  # For saving and loading preprocessed data
import torch.nn as nn
import torch
from senT_ce import CustomBert
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import logging

def train(dataset_name):
    # 1. 데이터셋 로드
    dataset = load_dataset('Linq-AI-Research/FinanceRAG', dataset_name)

    # 2. CSV 파일 로드
    qrels_df = pd.read_csv(f'gt/{dataset_name}_qrels.tsv', sep='\t', header=None, names=['query_id', 'corpus_id', 'score'])

    # 3. corpus와 queries 데이터셋에서 '_id'를 기준으로 텍스트 매핑
    corpus_df = pd.DataFrame(dataset['corpus'])
    queries_df = pd.DataFrame(dataset['queries'])

    query_id_to_text = pd.Series(queries_df.text.values, index=queries_df._id).to_dict()
    corpus_id_to_text = pd.Series(corpus_df.text.values, index=corpus_df._id).to_dict()

    # 4. 긍정 쌍 리스트 생성
    positive_pairs = []

    for _, row in qrels_df.iterrows():
        query_id = row['query_id']
        corpus_id = row['corpus_id']
        query_text = query_id_to_text.get(query_id, "")
        corpus_text = corpus_id_to_text.get(corpus_id, "")
        if query_text and corpus_text:
            positive_pairs.append((query_text, corpus_text))

    print(f"Total positive pairs: {len(positive_pairs)}")
    
    def get_negative_corpus(query_id):
        positive_corpus_ids = qrels_df[(qrels_df['query_id'] == query_id) & (qrels_df['score'] == 1)]['corpus_id'].tolist()
        negative_corpus = list(set(dataset['corpus']['_id']) - set(positive_corpus_ids))
        return set(negative_corpus)
    
    from torch.utils.data import Dataset
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
            input_examples.append({'label':1.0, 
                                   'query': query, 'doc': positive})#InputExample(texts=[query+"[CLS]"+positive], label=1.0))
            for j in range(batch_size):
                if i != j:
                    negative = batch[j]['positive']
                    # Add negative sample
                    input_examples.append({'label':0.0, 
                                           'query': query, 'doc': negative})#InputExample(texts=[query+"[CLS]"+negative], label=0.0))
        return input_examples
    
    # 7. Load the model
    # ce = CrossEncoder('Capreolus/bert-base-msmarco')
    # ce_model = ce.model
    # cb = CustomBert(ce_model.config)
    # cb.bert.load_state_dict(ce_model.bert.state_dict())
    # cb.classifier = nn.Linear(768, 1)

    # model = SentenceTransformerCL(modules=[cb], tokenizer_name='Capreolus/bert-base-msmarco')
        

    # 8. Create Dataset and DataLoader
    train_dataset = FinanceRAGDataset(positive_pairs)
    batch_size = 16  # Adjust as needed
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=in_batch_collate_fn)
    
    # 9. Define the loss function
    # train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=768, num_labels=1)
    
    # # 10. Fine-tuning settings
    # num_epochs = 20
    # warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    
    # # 11. Fine-tune the model
    # model.fit(
    #     train_objectives=[(train_dataloader, train_loss)],
    #     epochs=num_epochs,
    #     warmup_steps=warmup_steps,
    #     show_progress_bar=True,
    #     output_path=f'.models/combined/co-condenser-marco_e20'
    # )
    
    # # 12. Save the model
    # model.save(f'.models/combined/co-condenser-marco_e20')

    model = AutoModelForSequenceClassification.from_pretrained(
        "Capreolus/bert-base-msmarco"
    )
    tokenizer = AutoTokenizer.from_pretrained("Capreolus/bert-base-msmarco")
    # model = CustomBert(model.config)
    # model.load_state_dict(model.state_dict())
    # model.train()
    
    batchs = {'labels':[], 'queries':[], 'docs':[]}
    for batch in train_dataloader:
        batchs['labels'].extend([int(d['label']) for d in batch])
        batchs['queries'].extend([d['query'] for d in batch])
        batchs['docs'].extend([d['doc'] for d in batch])
        
    from datasets import Dataset
    import pyarrow as pa
    batchs = pa.table(batchs)
    batchs = Dataset(batchs)
    
    def tokenize_function(examples):
        queries = examples['queries']
        docs = examples['docs']
        
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []

        for query, doc in zip(queries, docs):
            # Query 토크나이징
            query_tokens = tokenizer(query, return_tensors='pt')
            q_len = len(query_tokens['input_ids'][0])
            
            doc_chunk_len = 512 - q_len #+ 1
            if doc_chunk_len <= 0:
                # query가 너무 길어서 doc chunk를 둘 공간이 없음.
                # 이런 경우 에러 처리하거나 query를 잘라내는 정책 필요
                raise ValueError("Query is too long to fit with any doc tokens into the max length.")
            
            # Document 토크나이징
            doc_tokens =  tokenizer(doc, 
                                    max_length = doc_chunk_len,
                                    return_tensors='pt',
                                    padding="max_length", 
                                    truncation=True,
                                    return_overflowing_tokens=True)
            q=1
            ds = doc_tokens['input_ids'].shape
            if ds[0] < q:
                extra_decoded = tokenizer(["[PAD]"]*(q-ds[0]), 
                                            max_length = doc_chunk_len,
                                            return_tensors='pt',
                                            padding="max_length", 
                                            truncation=True,
                                            return_overflowing_tokens=True)
                for key in doc_tokens.keys():
                    doc_tokens[key] = torch.cat([doc_tokens[key], extra_decoded[key]], dim=0)
            
            # for key in ['input_ids', 'attention_mask', 'token_type_ids']:
            #     doc_tokens[key] = doc_tokens[key][:,1:]
                
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            q_i = query_tokens['input_ids'].squeeze(0)
            q_a = query_tokens['attention_mask'].squeeze(0)
            q_t = query_tokens['token_type_ids'].squeeze(0)

            for k in range(q):                
                d_i = doc_tokens['input_ids'][k]
                d_a = doc_tokens['attention_mask'][k]
                d_t = doc_tokens['token_type_ids'][k]
                assert len(d_i) + len(q_i) == 512
                input_ids_list.append(torch.concat([q_i, d_i], dim=0))
                attention_mask_list.append(torch.concat([q_a, d_a], dim=0))
                token_type_ids_list.append(torch.concat([q_t, d_t], dim=0))
                
            # 한 예제 당 4개 시퀀스를 각각 tensor로 변환
            # input_ids_tensor = torch.concat(input_ids_list, dim=0)
            # attention_mask_tensor = torch.concat(attention_mask_list, dim=0)
            # token_type_ids_tensor = torch.concat(token_type_ids_list, dim=0)
        
            input_ids_batch.append(input_ids_list[0])
            attention_mask_batch.append(attention_mask_list[0])
            token_type_ids_batch.append(token_type_ids_list[0])

        # batch 모두 tensor로 스택 (batch_size, 2048) 형태
        return {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "attention_mask": torch.stack(attention_mask_batch, dim=0),
            "token_type_ids": torch.stack(token_type_ids_batch, dim=0)
        }

    tokenized_datasets = batchs.map(tokenize_function, batched=True)
    
    import numpy as np
    import evaluate
    metric = evaluate.load("recall")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir="ce_finetune",
        learning_rate=2e-5,
        per_device_train_batch_size=24,
        num_train_epochs=80,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        logging_dir="./logs",          # 로그를 저장할 디렉토리
        logging_steps=100,              # 매 10 스텝마다 로그 출력
        logging_strategy="steps",      # 'steps' 전략 사용
        report_to=["wandb"],          # 로그를 콘솔에 출력
        # report_to=["tensorboard", "wandb", "stdout"], # 원하는 대로 설정 가능
        # 기타 설정...
    )
    
    # 로그 레벨 설정 (선택 사항)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    # 11. Define metrics
    import numpy as np
    import evaluate
    metric = evaluate.load("recall")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # 12. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        compute_metrics=compute_metrics,
        # callbacks=[...], # 필요 시 콜백 추가
    )
    
    # 13. Start Training
    trainer.train()
    torch.save(model.state_dict(), f'models/{dataset_name}/bert_firstp.pt')

if __name__ == "__main__":
    # os.environ['WANDB_DISABLED'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
    for task_name in task_names:
        print(task_name)
        train(task_name)
        torch.cuda.empty_cache()