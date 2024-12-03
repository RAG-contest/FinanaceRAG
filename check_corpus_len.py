from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

dataset_name = "FinQA"
tokenizer = AutoTokenizer.from_pretrained(f".models/{dataset_name}/co-condenser-marco")
dataset = load_dataset('Linq-AI-Research/FinanceRAG', dataset_name)

corpus_df = pd.DataFrame(dataset['corpus'])
queries_df = pd.DataFrame(dataset['queries'])

query_id_to_text = pd.Series(queries_df.text.values, index=queries_df._id).to_dict()
corpus_id_to_text = pd.Series(corpus_df.text.values, index=corpus_df._id).to_dict()

for corpus_id in corpus_df['_id']:
    text = corpus_id_to_text.get(corpus_id, "")
    tokenized = tokenizer(text)
    
    if len(tokenized['input_ids']) >= 512:
        print(len(text), len(tokenized['input_ids']))