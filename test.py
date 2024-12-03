from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize
import nltk
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import random_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델과 토크나이저 불러오기
model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Query와 Corpus 불러오기
ds = load_dataset("Linq-AI-Research/FinanceRAG", "FinDER")
queries = ds['queries']['text']
corpus = ds['corpus']['text']

# Define the preprocessing function for tokenization
def preprocess_function(examples):
    print(sent_tokenize(examples["text"]))
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = "[SEP]".join(sent_tokenize(text))
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: val.squeeze(0) for key, val in encoded.items()}

# Dataset 생성
query_dataset = TextDataset(queries, tokenizer)
corpus_dataset = TextDataset(corpus, tokenizer)

# Dataloader 생성
batch_size = 32  # 원하는 배치 크기 설정
query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
corpus_dataloader = DataLoader(corpus_dataset, batch_size=batch_size, shuffle=False)

# 쿼리와 코퍼스의 임베딩 생성
query_embeddings = []
corpus_embeddings = []

with torch.no_grad():  # 그래디언트 계산 비활성화
    # 쿼리 처리
    for batch in tqdm(query_dataloader):
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        query_embeddings.append(outputs.pooler_output.cpu())

    # 코퍼스 처리
    for batch in tqdm(corpus_dataloader):
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        corpus_embeddings.append(outputs.pooler_output.cpu())

# 임베딩 합치기
query_embeddings = torch.cat(query_embeddings, dim=0)
corpus_embeddings = torch.cat(corpus_embeddings, dim=0)

# Normalize the embeddings
query_embeddings = normalize(query_embeddings, dim=1)
corpus_embeddings = normalize(corpus_embeddings, dim=1)

# Calculate cosine similarity
cosine_similarities = torch.mm(query_embeddings, corpus_embeddings.T)

qid = ds['queries']['_id']
cid = ds['corpus']['_id']

# Top-10 retrieval
k = 10  # 상위 10개의 결과 추출
top_k_values, top_k_indices = torch.topk(cosine_similarities, k=k, dim=1)

# 결과를 저장할 딕셔너리
results = {}

# 딕셔너리 구성
for query_idx, (indices, scores) in enumerate(zip(top_k_indices, top_k_values)):
    query_id = qid[query_idx]  # 쿼리 ID
    results[query_id] = {
        cid[corpus_idx]: score.item() for corpus_idx, score in zip(indices, scores)
    }

import pandas as pd
from financerag.tasks import FinDER

# FinDER 작업 초기화
finder_task = FinDER()

# 답변 레이블의 30%가 포함된 TSV 파일 로드
df = pd.read_csv('gt/FinDER_qrels.tsv', sep='\t')

# TSV 데이터를 평가를 위한 사전 형식으로 변환
qrels_dict = df.groupby('query_id').apply(lambda x: dict(zip(x['corpus_id'], x['score']))).to_dict()

# 검색 또는 재정렬 결과가 `results` 변수에 저장된 경우
# Recall, Precision, MAP, nDCG와 같은 다양한 지표로 모델 평가
print(finder_task.evaluate(qrels_dict, results, [1, 5, 10]))

# 평가 결과를 출력합니다 (즉, `Recall`, `Precision`, `MAP`, `nDCG`)