import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import losses, InputExample
from sentence_transformers import models
from transformersCL import SentenceTransformerCL
import random
from tqdm import tqdm

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
            input_examples.append(InputExample(texts=[query, positive], label=1.0))
            for j in range(batch_size):
                if i != j:
                    negative = batch[j]['positive']
                    # 부정 샘플 추가
                    input_examples.append(InputExample(texts=[query, negative], label=0.0))
        return input_examples

    # 7. 모델 로드
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformerCL(model_name)
    
    # 8. 데이터셋 및 DataLoader 생성
    train_dataset = FinanceRAGDataset(positive_pairs)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=in_batch_collate_fn)

    # 9. 손실 함수 정의
    train_loss = losses.ContrastiveLoss(model=model)

    # 10. 파인튜닝 설정
    num_epochs = 3
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

    # 11. 모델 파인튜닝
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=f'./all-MiniLM-L6-v2_{dataset_name}'
    )

    # 12. 모델 저장
    model.save(f'./all-MiniLM-L6-v2_{dataset_name}')

if __name__ == "__main__":
    task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
    for task_name in task_names:
        train(task_name)