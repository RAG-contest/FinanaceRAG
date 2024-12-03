import copy
import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig, BertPooler
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from bert_mores import MORES_BertLayer  # Ensure this is correctly implemented
# from arguments import ModelArguments, DataArguments, MORESTrainingArguments as TrainingArguments  # Removed

logger = logging.getLogger(__name__)


class MORESSym(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        n_ibs: int = 6,  # Default number of interaction blocks
        use_pooler: bool = True,
        copy_weight_to_ib: bool = False,
        hidden_size: int = None,  # Optional: specify if different from BERT's hidden size
        num_labels: int = 2,  # For classification tasks
    ):
        """
        Simplified MORESSym model without using argument instances.

        Args:
            bert (BertModel): Pretrained BERT model.
            n_ibs (int): Number of interaction blocks.
            use_pooler (bool): Whether to use the BERT pooler.
            copy_weight_to_ib (bool): Whether to copy weights to interaction blocks.
            hidden_size (int, optional): Hidden size for projection. Defaults to BERT's hidden size.
            num_labels (int): Number of classification labels.
        """
        super().__init__()
        self.bert = bert
        config_m = copy.deepcopy(bert.config)
        config_m.is_decoder = True
        config_m.add_cross_attention = True

        self.interaction_module = nn.ModuleList(
            [MORES_BertLayer(config_m) for _ in range(n_ibs)]
        )
        self.interaction_module.apply(bert._init_weights)

        if copy_weight_to_ib:
            # Assuming BERT-base with 12 layers
            for i in range(n_ibs):
                layer_idx = -n_ibs + i  # Start copying from the last n_ibs layers
                self.interaction_module[i].attention = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].attention
                )
                self.interaction_module[i].crossattention = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].attention
                )
                self.interaction_module[i].intermediate = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].intermediate
                )
                self.interaction_module[i].output = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].output
                )

        self.use_pooler = use_pooler
        if use_pooler:
            self.pooler = BertPooler(config_m)

        projection_hidden_size = hidden_size if hidden_size else config_m.hidden_size
        self.proj = nn.Linear(projection_hidden_size, num_labels)
        self.cross_entropy = CrossEntropyLoss(reduction='mean')

        # Store parameters for potential future use or debugging
        self.n_ibs = n_ibs
        self.copy_weight_to_ib = copy_weight_to_ib
        self.num_labels = num_labels

    def forward(self, qry, doc, labels=None):
        """
        Forward pass for the MORESSym model.

        Args:
            qry (dict): Query inputs with keys like 'input_ids', 'attention_mask', etc.
            doc (dict): Document inputs with similar keys.
            labels (torch.Tensor, optional): Labels for classification. Defaults to None.

        Returns:
            SequenceClassifierOutput: Output containing loss, logits, and attentions.
        """
        qry_out = self._encode_query(qry)
        doc_out = self._encode_document(doc)

        self_mask = self.bert.get_extended_attention_mask(
            qry['attention_mask'], qry['attention_mask'].shape, qry['attention_mask'].device
        )
        cross_mask = self.bert.get_extended_attention_mask(
            doc['attention_mask'], doc['attention_mask'].shape, doc['attention_mask'].device
        )

        hidden_states = qry_out.last_hidden_state
        interaction_self_attention = ()
        interaction_cross_attention = ()

        for ib_layer in self.interaction_module:
            layer_outputs = ib_layer(
                hidden_states,
                attention_mask=self_mask,
                encoder_hidden_states=doc_out.last_hidden_state,
                encoder_attention_mask=cross_mask,
                output_attentions=self.bert.config.output_attentions,
            )
            hidden_states = layer_outputs[0]
            if self.bert.config.output_attentions:
                interaction_self_attention += (layer_outputs.attentions,)
                if hasattr(self.bert.config, 'add_cross_attention') and self.bert.config.add_cross_attention:
                    interaction_cross_attention += (layer_outputs.cross_attentions,)

        if self.use_pooler:
            cls_reps = self.pooler(hidden_states)
        else:
            cls_reps = hidden_states[:, 0]  # [CLS] token

        logits = self.proj(cls_reps)

        loss = self.cross_entropy(logits, labels) if self.training and labels is not None else None

        all_attentions = (
            doc_out.attentions,
            qry_out.attentions,
            interaction_self_attention,
            interaction_cross_attention
        ) if self.bert.config.output_attentions else None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=all_attentions,
        )

    def _encode_document(self, doc):
        return self.bert(**doc, return_dict=True)

    def _encode_query(self, qry):
        return self.bert(**qry, return_dict=True)


class MORES(MORESSym):
    def __init__(
        self,
        bert: BertModel,
        n_ibs: int = 6,
        use_pooler: bool = True,
        copy_weight_to_ib: bool = True,
        hidden_size: int = None,
        num_labels: int = 2,
    ):
        """
        Simplified MORES model inheriting from MORESSym.

        Args:
            bert (BertModel): Pretrained BERT model.
            n_ibs (int): Number of interaction blocks.
            use_pooler (bool): Whether to use the BERT pooler.
            copy_weight_to_ib (bool): Whether to copy weights to interaction blocks.
            hidden_size (int, optional): Hidden size for projection. Defaults to BERT's hidden size.
            num_labels (int): Number of classification labels.
        """
        super().__init__(
            bert=bert,
            n_ibs=n_ibs,
            use_pooler=use_pooler,
            copy_weight_to_ib=copy_weight_to_ib,
            hidden_size=hidden_size,
            num_labels=num_labels,
        )

        # Create a separate BERT model for queries
        self.q_bert = copy.deepcopy(bert)

        if copy_weight_to_ib:
            # Assuming BERT-base with 12 layers
            for i in range(n_ibs):
                layer_idx = 12 - n_ibs + i  # Adjust layer index as needed
                self.interaction_module[i].attention = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].attention
                )
                self.interaction_module[i].crossattention = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].attention
                )
                self.interaction_module[i].intermediate = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].intermediate
                )
                self.interaction_module[i].output = copy.deepcopy(
                    self.bert.encoder.layer[layer_idx].output
                )

        # Modify the query BERT encoder to exclude the last n_ibs layers
        self.q_bert.encoder.layer = nn.ModuleList(
            [self.q_bert.encoder.layer[i] for i in range(12 - n_ibs)]
        )

    def _encode_query(self, qry):
        return self.q_bert(**qry, return_dict=True)

    def _encode_document(self, doc):
        return self.bert(**doc, return_dict=True)
