from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F 
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from financerag.common import Encoder

class EncoderModel(nn.Module):
    def __init__(self, model_name):
        super(EncoderModel, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :]  # [CLS] 임베딩

    def forward(self, input_ids, attention_mask):
        emb = self.encode(input_ids, attention_mask)
        return emb

# Adopted by https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/sentence_bert.py
class AutoTransformerEncoder(Encoder):
    def __init__(
            self,
            model_name_or_path: Union[str, Tuple[str, str]],
            **kwargs
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(model_name_or_path, str):
            self.q_model = EncoderModel(model_name_or_path, **kwargs).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.doc_model = self.q_model
        else:
            raise TypeError

    def encode_queries(
            self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[np.ndarray, Tensor]:
        ret_tensor = None
        for i in tqdm(range(0, len(queries), batch_size)):
            batch_query = self.tokenizer(
                queries[i:i+batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            batch_query = self.q_model(input_ids=batch_query['input_ids'], attention_mask=batch_query['attention_mask'])
            batch_query = F.normalize(batch_query, dim=1)
            if ret_tensor == None:
                ret_tensor = batch_query.detach().cpu()
            else:
                ret_tensor = torch.concat([ret_tensor, batch_query.detach().cpu()], dim=0)
                
        return ret_tensor.numpy()

    def encode_corpus(
            self,
            corpus: Union[
                List[Dict[Literal["title", "text"], str]],
                Dict[Literal["title", "text"], List],
            ],
            batch_size: int = 8,
            **kwargs
    ) -> Union[np.ndarray, Tensor]:
        ret_tensor = None
        for i in tqdm(range(0, len(corpus), batch_size)):
            batch_query = self.tokenizer(
                ["[SEP]".join(sent_tokenize(c['text'])) for c in corpus[i:i+batch_size]],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            batch_query = self.q_model(input_ids=batch_query['input_ids'], attention_mask=batch_query['attention_mask'])
            batch_query = F.normalize(batch_query, dim=1)
            
            if ret_tensor == None:
                ret_tensor = batch_query.detach().cpu()
            else:
                ret_tensor = torch.concat([ret_tensor, batch_query.detach().cpu()], dim=0)
                
        return ret_tensor