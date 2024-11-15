# FinDER for example.
# You can use other tasks such as `FinQA`, `TATQA`, etc.
from financerag.tasks import FinDER, FinQABench,FinanceBench, TATQA, FinQA, ConvFinQA, MultiHiertt
tasks = [FinDER(), FinQABench(), FinanceBench(), TATQA(), FinQA(), ConvFinQA(), MultiHiertt()]

from sentence_transformers import SentenceTransformer
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval
from sentence_transformers import CrossEncoder
from financerag.rerank import CrossEncoderReranker
import torch
import os
import pandas as pd

reranked_results = []
bi_model_names_per_task = ['outputs/models/FinDER/finance_bi_encoder_e10_b17',# FinDER
                        'outputs/models/FinQABench/finance_bi_encoder_e10_b20',# FinQABench
                        'outputs/models/FinanceBench/finance_bi_encoder_e10_b20',# FinanceBench
                        'outputs/models/TATQA/finance_bi_encoder_e10_b20',# TATQA
                        'outputs/models/FinQA/finance_bi_encoder_e20_b20',# FinQA
                        'outputs/models/ConvFinQA/finance_bi_encoder_e20_b20',# ConvFinQA
                        'outputs/models/MultiHiertt/finance_bi_encoder_e10_b20' # MultiHiertt
                        ]
ce_model_names_per_task = ['outputs/models/FinDER/finance_cross_encoder_model_e10_b10',# FinDER
                        'outputs/models/FinQABench/finance_cross_encoder_model_e10_b12',# FinQABench
                        'outputs/models/FinanceBench/finance_cross_encoder_model_e10_b12',# FinanceBench
                        'outputs/models/TATQA/finance_cross_encoder_model_e10_b12',# TATQA
                        'outputs/models/FinQA/finance_cross_encoder_model_e20_b12',# FinQA
                        'outputs/models/ConvFinQA/finance_cross_encoder_model_e20_b12',# ConvFinQA
                        'outputs/models/MultiHiertt/finance_cross_encoder_model_e10_b12' # MultiHiertt
                        ]
                        
for task, bi_model_name_task, ce_model_name_task in zip(tasks, bi_model_names_per_task, ce_model_names_per_task):
  # model = SentenceTransformer('intfloat/e5-large-v2')
  # We need to put prefix for e5 models.
  # For more details, see Arxiv paper https://arxiv.org/abs/2212.03533
  encoder_model = SentenceTransformerEncoder(
      model_name_or_path=bi_model_name_task,
      # q_model=model,
      # doc_model=model,
      query_prompt='query: ',
      doc_prompt='passage: '
  )
  retriever = DenseRetrieval(model=encoder_model)

  # Retrieve relevant documents
  print("retrieval")
  results = task.retrieve(retriever=retriever, show_progress_bar=True, top_k=None)
  # results = tfidf_retrieval()

  # Rerank the results
  # model_names = ['cross-encoder/ms-marco-MiniLM-L-12-v2']
  # model = EnsembledCrossEncoder(model_names, device='cuda')

  model = CrossEncoder(ce_model_name_task)
  reranker = CrossEncoderReranker(model)
  print("rearanking")
  reranked_results.append(task.rerank(reranker, results, top_k=5000, batch_size=64, show_progress_bar=True))
  task.save_results(output_dir='outputs/')

tasks = [FinDER(), FinQABench(), FinanceBench(), TATQA(), FinQA(), ConvFinQA(), MultiHiertt()]
task_names = ["FinDER", "FinQABench", "FinanceBench", "TATQA", "FinQA", "ConvFinQA", "MultiHiertt"]
for i, task in enumerate(tasks):
  task_name = task_names[i]
  # 답변 레이블의 30%가 포함된 TSV 파일 로드
  df = pd.read_csv(f'gt/{task_name}_qrels.tsv', sep='\t')

  # TSV 데이터를 평가를 위한 사전 형식으로 변환
  qrels_dict = df.groupby('query_id').apply(lambda x: dict(zip(x['corpus_id'], x['score']))).to_dict()
  # 검색 또는 재정렬 결과가 `results` 변수에 저장된 경우
  # Recall, Precision, MAP, nDCG와 같은 다양한 지표로 모델 평가
  print(task.evaluate(qrels_dict, reranked_results[i], [1, 5, 10]))


# Define the path to the 'outputs' directory
outputs_dir = 'outputs'
all_data = []  # List to hold data from each CSV
# Walk through the 'outputs' directory and subdirectories
for root, dirs, files in os.walk(outputs_dir):
    for file in files:
        if file == 'results.csv':  # Look for 'result.csv' files
            file_path = os.path.join(root, file)
            # Read each CSV and append to the list
            df = pd.read_csv(file_path)
            all_data.append(df)

# Concatenate all data into a single DataFrame
merged_df = pd.concat(all_data, ignore_index=True)

# Save the merged data to a new CSV file
merged_df.to_csv('merged_result.csv', index=False)
print("Merged CSV saved as 'merged_result.csv'")