import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 데이터셋 로드
dataset = load_dataset('Linq-AI-Research/FinanceRAG', 'MultiHiertt')
corpus_dataset = dataset['corpus']
queries_dataset = dataset['queries']

# 2. CSV 파일 로드
qrels_df = pd.read_csv('gt/MultiHiertt_qrels.tsv', sep='\t', header=None, names=['query_id', 'corpus_id', 'score'])

# 3. corpus와 queries 데이터셋을 DataFrame으로 변환
corpus_df = pd.DataFrame(corpus_dataset)
queries_df = pd.DataFrame(queries_dataset)

# ID를 텍스트로 매핑하기 위한 딕셔너리 생성
assert '_id' in queries_df.columns, "queries 데이터셋에 '_id' 컬럼이 없습니다."
assert '_id' in corpus_df.columns, "corpus 데이터셋에 '_id' 컬럼이 없습니다."

query_id_to_text = pd.Series(queries_df.text.values, index=queries_df._id).to_dict()
corpus_id_to_text = pd.Series(corpus_df.text.values, index=corpus_df._id).to_dict()

# 4. 긍정 쌍 리스트 생성 (벡터화된 접근)
mask = qrels_df['query_id'].isin(query_id_to_text) & qrels_df['corpus_id'].isin(corpus_id_to_text)
filtered_qrels = qrels_df[mask]

positive_pairs = list(zip(
    filtered_qrels['query_id'].map(query_id_to_text),
    filtered_qrels['corpus_id'].map(corpus_id_to_text)
))

print(f"Total positive pairs: {len(positive_pairs)}")

# 5. PyTorch Dataset 클래스 정의 (필요 시)
class FinanceRAGDataset(Dataset):
    def __init__(self, positive_pairs):
        self.positive_pairs = positive_pairs

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        query, positive = self.positive_pairs[idx]
        return query, positive

# 6. 듀얼 인코더 모델 정의
class DualEncoderModel(nn.Module):
    def __init__(self, model_name):
        super(DualEncoderModel, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]  # [CLS] 임베딩

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        emb1 = self.encode(input_ids1, attention_mask1)
        emb2 = self.encode(input_ids2, attention_mask2)
        return emb1, emb2

# 8. 모델 및 토크나이저 로드
fine_tuned_model_path = './fine-tuned-co-condenser-marco-dual-encoder'
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
dual_encoder = DualEncoderModel(fine_tuned_model_path)

# GPU 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dual_encoder.to(device)
dual_encoder.eval()

# 9. 임베딩 생성 함수 정의
def encode_texts(texts, tokenizer, model, device, batch_size=32):
    embeddings = []
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="Encoding Texts"):
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            emb = model.encode(encoded['input_ids'], encoded['attention_mask'])
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# 코퍼스 텍스트 리스트
corpus_texts = corpus_df['text'].tolist()

print("Encoding Corpus...")
corpus_embeddings = encode_texts(corpus_texts, tokenizer, dual_encoder, device)
print(f"Corpus Embeddings Shape: {corpus_embeddings.shape}")

# 쿼리 텍스트 리스트
query_texts = queries_df['text'].tolist()

print("Encoding Queries...")
query_embeddings = encode_texts(query_texts, tokenizer, dual_encoder, device)
print(f"Query Embeddings Shape: {query_embeddings.shape}")

# 10. 유사도 계산 및 상위 k 문서 검색 (배치 단위로 최적화)
def compute_top_k(query_embeddings, corpus_embeddings, k_values=[1,5,10], batch_size=128):
    top_k_indices = {k: [] for k in k_values}
    num_queries = query_embeddings.shape[0]
    for start in tqdm(range(0, num_queries, batch_size), desc="Computing Top-K"):
        end = min(start + batch_size, num_queries)
        batch_queries = query_embeddings[start:end]
        # 코사인 유사도는 정규화된 임베딩 사용 시 내적과 동일
        # 따라서, 미리 정규화하여 내적을 코사인 유사도로 사용할 수 있음
        batch_queries_norm = batch_queries / np.linalg.norm(batch_queries, axis=1, keepdims=True)
        corpus_embeddings_norm = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        similarity = np.dot(batch_queries_norm, corpus_embeddings_norm.T)
        for k in k_values:
            top_k = np.argsort(-similarity, axis=1)[:, :k]
            top_k_indices[k].extend(top_k)
    # Convert lists to numpy arrays
    for k in k_values:
        top_k_indices[k] = np.array(top_k_indices[k])
    return top_k_indices

print("Computing Top-K indices...")
k_values = [1, 5, 10]
top_k_indices = compute_top_k(query_embeddings, corpus_embeddings, k_values)

# 11. NDCG 계산 함수 정의 (이진 관련성)
def dcg_at_k_binary(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    gains = relevances / np.log2(np.arange(2, relevances.size + 2))
    return np.sum(gains)

def ndcg_at_k_binary(relevant_docs, retrieved_docs, k):
    relevances = [1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]
    dcg = dcg_at_k_binary(relevances, k)
    # 이상적인 DCG (모든 관련 문서가 상위에 위치한 경우)
    ideal_relevances = [1] * min(len(relevant_docs), k)
    dcg_max = dcg_at_k_binary(ideal_relevances, k)
    return dcg / dcg_max if dcg_max > 0 else 0.0

# 12. Ground Truth 매핑
# corpus_df의 인덱스는 0부터 시작하므로, '_id'를 인덱스로 매핑
corpus_id_to_index = pd.Series(corpus_df.index.values, index=corpus_df['_id']).to_dict()
query_id_to_index = pd.Series(queries_df.index.values, index=queries_df['_id']).to_dict()

# qrels_df를 통해 query_idx -> set(corpus_idx) 매핑
query_to_relevant = {}

for _, row in filtered_qrels.iterrows():
    query_id = row['query_id']
    corpus_id = row['corpus_id']
    query_idx = query_id_to_index.get(query_id, None)
    corpus_idx = corpus_id_to_index.get(corpus_id, None)
    if query_idx is not None and corpus_idx is not None:
        if query_idx not in query_to_relevant:
            query_to_relevant[query_idx] = set()
        query_to_relevant[query_idx].add(corpus_idx)

print(f"Total queries with relevant documents: {len(query_to_relevant)}")

# 13. NDCG 계산
ndcg_at_1 = []
ndcg_at_5 = []
ndcg_at_10 = []

print("Calculating NDCG metrics...")

for q_idx in tqdm(range(len(query_texts)), desc="Calculating NDCG"):
    relevant_docs = query_to_relevant.get(q_idx, set())
    if not relevant_docs:
        ndcg_at_1.append(0.0)
        ndcg_at_5.append(0.0)
        ndcg_at_10.append(0.0)
        continue

    retrieved_docs = {}
    for k in k_values:
        retrieved_docs[k] = top_k_indices[k][q_idx]

    ndcg_at_1.append(ndcg_at_k_binary(relevant_docs, retrieved_docs[1], 1))
    ndcg_at_5.append(ndcg_at_k_binary(relevant_docs, retrieved_docs[5], 5))
    ndcg_at_10.append(ndcg_at_k_binary(relevant_docs, retrieved_docs[10], 10))

# 평균 NDCG 계산
avg_ndcg_1 = np.mean(ndcg_at_1)
avg_ndcg_5 = np.mean(ndcg_at_5)
avg_ndcg_10 = np.mean(ndcg_at_10)

print(f"Average NDCG@1: {avg_ndcg_1:.4f}")
print(f"Average NDCG@5: {avg_ndcg_5:.4f}")
print(f"Average NDCG@10: {avg_ndcg_10:.4f}")
