from torch.utils.data import Dataset
from sentence_transformers import InputExample

class FinanceDataset(Dataset):
    def __init__(self, positive_pairs, all_corpus_ids, get_id_text, mode='train',):
        self.positive_pairs = positive_pairs
        self.all_corpus_ids = all_corpus_ids
        self.map_fn = get_id_text
        self.mode = mode

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
        
        # Return positive pair
        # positive_example = InputExample(texts=[query_text, positive_corpus_text], label=1.0)
        
        return query_text, positive_corpus_text, query_id, positive_corpus_id