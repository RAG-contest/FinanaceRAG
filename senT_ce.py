from sentence_transformers import CrossEncoder
from transformers import BertForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss, BCELoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput

class CustomBert(BertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained('Capreolus/bert-base-msmarco')
        def f(*args, **kwargs):
            for text in args[0]:
                if type(text) != type("asdf"):
                    print(text, type(text))
            kwargs['return_tensors'] = 'pt'
            kwargs['truncation'] = True
            kwargs['padding'] = True
            return self.tokenizer(*args, **kwargs)
        self.tokenize = f
        
    # def forward(self, *args, **kwargs):
    #     label = kwargs['labels']
    #     input_ids = kwargs['input_ids']
    #     attention_mask = kwargs['attention_mask']
        
    #     batch_size = input_ids.shape[0]
    #     input_ids = input_ids.view(batch_size, 4, 512)
    #     attention_mask = attention_mask.view(batch_size, 4, 512)

    #     score = []
    #     loss = []
    #     for k in range(4):
    #         k_input_ids = input_ids[:,k,:]
    #         k_attention_mask = attention_mask[:,k,:]
    #         # print(label.shape)
    #         out = super().forward(input_ids=k_input_ids, attention_mask=k_attention_mask)
    #         # print(out.logits.shape, out.loss.shape, out.loss)
    #         score.append(out.logits)
    #     score = torch.stack(score, dim=1)
    #     # print(score.shape, loss.shape)
    #     score, indicies = torch.max(score[:,:,1], dim=1)
    #     score = score[torch.arange(batch_size),indicies]
    #     criterion = nn.CrossEntropyLoss()
    #     loss = criterion(score, label.squeeze(1).long())
    #     return loss, score
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        logits = []
        batch_size = input_ids.shape[0]
        for k in range(4):
            k_input_ids = input_ids.view(batch_size,4,-1)[:,k,:]
            k_attention_mask = attention_mask.view(batch_size,4,-1)[:,k,:]
            k_token_type_ids = token_type_ids.view(batch_size,4,-1)[:,k,:]
            outputs = self.bert(
                k_input_ids,
                attention_mask=k_attention_mask,
                token_type_ids=k_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits.append(self.classifier(pooled_output))
        logits = torch.stack(logits, dim=1)
        logits, _ = torch.max(logits, dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    