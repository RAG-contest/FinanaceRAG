import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, models
import torch
import torch.nn as nn
from sentence_transformers import losses, InputExample
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

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
        def __init__(self, positive_pairs):
            self.positive_pairs = positive_pairs

        def __len__(self):
            return len(self.positive_pairs)

        def __getitem__(self, idx):
            query, positive = self.positive_pairs[idx]
            return {'query': query, 'positive': positive}

    # 6. 커스텀 collate_fn 정의
    def in_batch_collate_fn(batch):
        """
        배치 내의 모든 positive pair를 negative pair로 사용하는 collate_fn
        """
        input_examples = []
        batch_size = len(batch)
        for i in range(batch_size):
            query = batch[i]['query']
            positive = batch[i]['positive']
            # 긍정 샘플 추가
            input_examples.append(InputExample(texts=[query, positive], label=1))
            for j in range(batch_size):
                if batch[i]['query'] != batch[j]['query']:
                    negative = batch[j]['positive']
                    # 부정 샘플 추가
                    input_examples.append(InputExample(texts=[query, negative], label=-1))

        return input_examples

    # 7. 듀얼 인코더 모델 정의
    class DualEncoderModel(nn.Module):
        def __init__(self, model_name):
            super(DualEncoderModel, self).__init__()
            self.model = models.Transformer(model_name)
            self.pooling = models.Pooling(self.model.get_word_embedding_dimension(), "cls")
            self.linear = nn.Linear(self.model.get_word_embedding_dimension(), self.model.get_word_embedding_dimension())
            self.cos = nn.CosineSimilarity(dim=1)
            
        def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
            # 첫 번째 입력 (쿼리) 인코딩
            outputs1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1, output_hidden_states=True)
            emb1 = self.linear(outputs1)  # [CLS] 토큰 임베딩
            
            # 두 번째 입력 (코퍼스) 인코딩
            outputs2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2, output_hidden_states=True)
            emb2 = self.linear(outputs2)  # [CLS] 토큰 임베

            return emb1, emb2

    # 9. 데이터로더 및 옵티마이저 설정
    train_dataset = FinanceRAGDataset(positive_pairs)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=6,
        collate_fn=in_batch_collate_fn
    )

    model_name = 'Luyu/co-condenser-marco'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dual_encoder = DualEncoderModel(model_name)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.AdamW(dual_encoder.parameters(), lr=2e-5)

    # 10. 트레이닝 루프 구현
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dual_encoder.to(device)

    epochs = 5
    for epoch in range(epochs):
        dual_encoder.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            # InputExamples 리스트에서 쿼리와 패시지 추출
            queries = [example.texts[0] for example in batch]
            passages = [example.texts[1] for example in batch]
            labels = torch.tensor([example.label for example in batch]).float().to(device)
            
            # 토크나이저로 인코딩
            encoded_queries = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            encoded_passages = tokenizer(
                ["[SEP]".join(sent_tokenize(p)) for p in passages],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # 임베딩 추출
            emb1, emb2 = dual_encoder(
                input_ids1=encoded_queries['input_ids'],
                attention_mask1=encoded_queries['attention_mask'],
                input_ids2=encoded_passages['input_ids'],
                attention_mask2=encoded_passages['attention_mask']
            )
            
            # 손실 계산
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss}")

    # 11. 모델 저장
    dual_encoder.model.save_pretrained(f'outputs/models/{dataset_name}/fine-tuned-co-condenser-marco-dual-encoder')
    tokenizer.save_pretrained(f'outputs/models/{dataset_name}/fine-tuned-co-condenser-marco-dual-encoder')

if __name__ == "__main__":
    train("MultiHiertt")