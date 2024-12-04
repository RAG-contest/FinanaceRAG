from torch.utils.data import Dataset
from sentence_transformers import InputExample

class FinanceDataset(Dataset):
    def __init__(self, positive_pairs, all_corpus_ids, get_id_text, mode='train', tokenizer=None):
        self.positive_pairs = positive_pairs
        self.all_corpus_ids = all_corpus_ids
        self.map_fn = get_id_text
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        query_id, positive_corpus_id = self.positive_pairs[idx]
        
        # Positive sample
        query_text = self.map_fn('queries', query_id)
        positive_corpus_text = self.map_fn('corpus', positive_corpus_id)
        if query_text is None or positive_corpus_text is None:
            # 텍스트가 없으면 다음 아이템으로 넘어갑니다
            return self.__getitem__((idx + 1) % len(self))
        
        if self.tokenizer is not None:
            positive_chunks = []
            positive_corpus_tokenized=self.tokenizer(positive_corpus_text,
                return_overflowing_tokens=True,
                padding=True,
                truncation=True,
                max_length=512,
                stride=0,
                return_tensors='pt'
              )['input_ids']
            for chunk in positive_corpus_tokenized:
                positive_chunks.append(self.tokenizer.decode(chunk))
            if len(positive_chunks) < 4:
                positive_chunks.extend(["[PAD]"]*(4-len(positive_chunks)))
            positive_chunks = positive_chunks[:4] # 4 from ANCE
        
            return query_text, positive_chunks, query_id, positive_corpus_id
        
        return query_text, positive_corpus_text, query_id, positive_corpus_id