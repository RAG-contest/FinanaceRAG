import pandas as pd
from datasets import load_dataset, DatasetDict

import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import losses, InputExample, models
from transformers import AutoTokenizer
from transformersCL import SentenceTransformerCL
import random
from tqdm import tqdm

from transformers import BertTokenizer

def split_text_into_chunks_with_special_tokens(text, tokenizer, max_length=512):    
    special_tokens_count = 2
    max_tokens = max_length - special_tokens_count
    
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        if i >= 4:
            break
        chunk = tokens[i:i + max_tokens]
        # [CLS]와 [SEP] 토큰 추가
        chunk = [tokenizer.cls_token] + chunk + [tokenizer.sep_token]
        chunks.append(chunk)
    
    # 토큰을 ID로 변환 (모델 입력용)
    input_ids = [tokenizer.convert_tokens_to_ids(chunk) for chunk in chunks]
    
    return input_ids

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

    # 5. PyTorch Dataset 클래스 정의
    class FinanceRAGDataset(Dataset):
        def __init__(self, positive_pairs, model_name):
            self.positive_pairs = positive_pairs
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.positive_pairs = {q:[self.tokenizer.decode(c_) for c_ in split_text_into_chunks_with_special_tokens(c, self.tokenizer, 512)] for q, c in self.positive_pairs}
            
        def __len__(self):
            return len(self.positive_pairs.keys())

        def __getitem__(self, idx):
            query = list(self.positive_pairs.keys())[idx]
            positives = self.positive_pairs[query]
            if len(positives) < 4:
                positives.extend(["[PAD]" for _ in range(4-len(positives))])
            return {'query': query, 'positives': positives}

    class ContrastiveLoss():
        def __init__(self):
            self.margin = 1
            
        def __call__(self, emb1, emb2, labels):
            distances = 1 - F.cosine_similarity(emb1, emb2)
            losses = 0.5 * (
                labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
            )
            return losses.mean()

    # 6. 커스텀 collate_fn 정의
    def in_batch_collate_fn(batch):
        """
        배치 내의 모든 positive pair를 negative pair로 사용하는 collate_fn
        """
        input_examples = []
        batch_size = len(batch)
        for i in range(batch_size):
            query = batch[i]['query']
            positives = batch[i]['positives']
            # 긍정 샘플 추가
            input_examples.append(InputExample(texts=[query, positives], label=1.0))
            for j in range(batch_size):
                if i != j:
                    negatives = batch[j]['positives']
                    # 부정 샘플 추가
                    input_examples.append(InputExample(texts=[query, negatives], label=0.0))
        return input_examples

    # 7. 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'Luyu/co-condenser-marco'
    cocon_model = models.Transformer(model_name)
    pooling = models.Pooling(cocon_model.get_word_embedding_dimension(), "cls")
    model = SentenceTransformerCL(modules=[cocon_model, pooling], device=device)
    # model_name = f".models/{dataset_name}/co-condenser-marco"
    # model = SentenceTransformerCL(model_name)
    
    # 8. 데이터셋 및 DataLoader 생성
    train_dataset = FinanceRAGDataset(positive_pairs, model_name)
    if dataset_name == "ConvFinQA":
        batch_size=32
    else:
        batch_size=48
        
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=in_batch_collate_fn)

    epochs = 5
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    optimizer = Adam(model.parameters(), lr=2e-5)
    steps_per_epoch = len(train_dataloader)
    num_train_steps = int(steps_per_epoch * epochs)
    scheduler = model._get_scheduler(
            optimizer, scheduler="WarmupLinear", warmup_steps=warmup_steps, t_total=num_train_steps
        )
    criterion = ContrastiveLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            # InputExamples 리스트에서 쿼리와 패시지 추출
            queries = [example.texts[0] for example in batch]
            passages = []
            k = 4
            for i in range(k):
                print(len(batch[0].texts[1]), batch[0].texts[1])
                passages.append([example.texts[1][i] for example in batch])
            labels = torch.tensor([example.label for example in batch]).float().to(device)
            
            q_embed = model.encode(queries)
            
            p_embed = None
            for i in range(k):
                if p_embed == None:
                    p_embed = model.encode(passages[i])
                else:
                    p_embed = torch.max(p_embed, model.encode(passages[i]))
            
            optimizer.zero_grad()
            loss = criterion(q_embed, p_embed, labels)
            loss.backward()  # 손실에 대한 역전파
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] 완료. 평균 손실: {average_loss:.4f}')
    
    model.save(f'models/{dataset_name}/fakemp_co-condenser-marco')        

if __name__ == "__main__":
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
    for task_name in task_names:
        print(task_name)
        train(task_name)
        
