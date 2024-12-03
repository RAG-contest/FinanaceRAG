from mores_plus import MORES
from transformers import models, HfArgumentParser
from arguments import ModelArguments, DataArguments

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

bert_name = 'Luyu/co-condenser-marco'
cocon_bert = models.Transformer(model_name)
mores = MORES(cocon_bert)

from transformers import BertTokenizer, BertModel

# Initialize tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Initialize the MORES model with desired parameters
mores_model = MORES(
    bert=bert_model,
    n_ibs=6,
    use_pooler=True,
    copy_weight_to_ib=True,
    hidden_size=768,  # Typically 768 for BERT-base
    num_labels=2,      # Binary classification
)

# Prepare sample inputs
query = tokenizer("Sample query text", return_tensors='pt')
document = tokenizer("Sample document text", return_tensors='pt')
labels = torch.tensor([1]).unsqueeze(0)  # Example label

# Forward pass
outputs = mores_model(qry=query, doc=document, labels=labels)

# Access loss and logits
loss = outputs.loss
logits = outputs.logits
