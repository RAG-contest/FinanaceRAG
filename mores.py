import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

class InteractionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward):
        super(InteractionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, Q, D, Q_padding_mask=None, D_padding_mask=None):
        # Q: (query_seq_length, batch_size, embed_dim)
        # D: (doc_seq_length, batch_size, embed_dim)
        
        # Cross-attention from Q to D (Equation 5)
        Q_residual = Q
        Qx, _ = self.cross_attention(
            Q, D, D,
            key_padding_mask=D_padding_mask,  # Mask for D
            attn_mask=None
        )
        Qx = self.cross_attention_layer_norm(Qx + Q_residual)
        
        # Self-attention over Qx (Equation 6)
        Qx_residual = Qx
        Qself, _ = self.self_attention(
            Qx, Qx, Qx,
            key_padding_mask=Q_padding_mask,  # Mask for Qx
            attn_mask=None
        )
        Qself = self.self_attention_layer_norm(Qself + Qx_residual)
        
        # Feed-forward network with residual connection (Equation 7)
        Qself_residual = Qself
        Qffn = self.ffn(Qself)
        output = self.ffn_layer_norm(Qffn + Qself_residual)
        
        return output

class InteractionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, num_interaction_blocks):
        super(InteractionModule, self).__init__()
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(embed_dim, num_heads, dim_feedforward)
            for _ in range(num_interaction_blocks)
        ])
        self.cls_projection = nn.Linear(embed_dim, 1)  # Project [CLS] embedding to a score
    
    def forward(self, Q, D, Q_padding_mask=None, D_padding_mask=None):
        # Q: (batch_size, query_seq_length, embed_dim)
        # D: (batch_size, doc_seq_length, embed_dim)
        
        # Transpose to (seq_length, batch_size, embed_dim)
        Q = Q.transpose(0, 1)  # (query_seq_length, batch_size, embed_dim)
        D = D.transpose(0, 1)  # (doc_seq_length, batch_size, embed_dim)
        
        H_IB = Q
        for ib in self.interaction_blocks:
            H_IB = ib(H_IB, D, Q_padding_mask=Q_padding_mask, D_padding_mask=D_padding_mask)
        
        # Transpose back to (batch_size, query_seq_length, embed_dim)
        H_IB = H_IB.transpose(0, 1)
        
        # Get the [CLS] token embedding (assumed to be at position 0)
        CLS_embedding = H_IB[:, 0, :]  # (batch_size, embed_dim)
        score = self.cls_projection(CLS_embedding).squeeze(-1)  # (batch_size)
        return score

class MORES(nn.Module):
    def __init__(self, model_name, num_heads, dim_feedforward, num_interaction_blocks):
        super(MORES, self).__init__()
        # Load the SentenceTransformer model
        self.query_transformer = AutoModelForMaskedLM.from_pretrained(model_name).model.encoder  # Transformer for queries
        self.doc_transformer = AutoModelForMaskedLM.from_pretrained(model_name).model.encoder    # Transformer for documents
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_dim = self.query_transformer.config.hidden_size
        
        # Freeze the parameters of the transformers
        # for param in self.query_transformer.parameters():
        #     param.requires_grad = False
        # for param in self.doc_transformer.parameters():
        #     param.requires_grad = False
            
        # Interaction Module
        self.interaction_module = InteractionModule(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_interaction_blocks=num_interaction_blocks
        )

    def get_token_embeddings(self, transformer_model, input_texts, doc=False):
        device = next(self.parameters()).device

        if doc:
            # Process a list of very long documents
            all_token_embeddings = []
            all_attention_masks = []
            max_seq_length = 0  # To keep track of the maximum sequence length

            for text in input_texts:
                # Tokenize the text with overflowing tokens
                encoded_inputs = self.tokenizer(
                    text,
                    return_overflowing_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    stride=0,
                    return_tensors='pt'
                )

                # Extract input_ids and attention_masks for all chunks
                input_ids_list = encoded_inputs['input_ids']  # List of tensors
                attention_mask_list = encoded_inputs['attention_mask']  # List of tensors

                chunk_embeddings = []
                chunk_attention_masks = []
                for input_ids_chunk, attention_mask_chunk in zip(input_ids_list, attention_mask_list):
                    # Add batch dimension
                    input_ids_chunk = input_ids_chunk.unsqueeze(0)  # (1, chunk_seq_length)
                    attention_mask_chunk = attention_mask_chunk.unsqueeze(0)  # (1, chunk_seq_length)

                    # Move inputs to the device
                    input_ids_chunk = input_ids_chunk.to(device)
                    attention_mask_chunk = attention_mask_chunk.to(device)

                    # Get token embeddings from the transformer
                    outputs = transformer_model(
                        input_ids=input_ids_chunk,
                        attention_mask=attention_mask_chunk,
                    )
                    
                    # Last hidden state is the token embeddings
                    token_embeddings_chunk = outputs.last_hidden_state  # (1, chunk_seq_length, embed_dim)

                    chunk_embeddings.append(token_embeddings_chunk)
                    chunk_attention_masks.append(attention_mask_chunk)

                # Concatenate chunks along the sequence dimension
                token_embeddings = torch.cat(chunk_embeddings, dim=1)  # (1, total_seq_length, embed_dim)
                attention_mask = torch.cat(chunk_attention_masks, dim=1)  # (1, total_seq_length)

                seq_length = token_embeddings.size(1)
                if seq_length > max_seq_length:
                    max_seq_length = seq_length

                all_token_embeddings.append(token_embeddings)
                all_attention_masks.append(attention_mask)

            # Pad all token_embeddings and attention_masks to the max_seq_length
            padded_token_embeddings = []
            padded_attention_masks = []
            for token_embeddings, attention_mask in zip(all_token_embeddings, all_attention_masks):
                seq_length = token_embeddings.size(1)
                pad_length = max_seq_length - seq_length

                if pad_length > 0:
                    # Pad token embeddings with zeros
                    pad_tensor = torch.zeros((1, pad_length, token_embeddings.size(2)), device=device)
                    token_embeddings = torch.cat([token_embeddings, pad_tensor], dim=1)

                    # Pad attention mask with zeros
                    pad_mask = torch.zeros((1, pad_length), device=device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

                padded_token_embeddings.append(token_embeddings)
                padded_attention_masks.append(attention_mask)

            # Stack them into tensors
            token_embeddings = torch.cat(padded_token_embeddings, dim=0)  # (batch_size, max_seq_length, embed_dim)
            attention_mask = torch.cat(padded_attention_masks, dim=0)  # (batch_size, max_seq_length)

        else:
            # Handle queries or shorter texts
            encoded_input = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = encoded_input['input_ids']  # (batch_size, seq_length)
            attention_mask = encoded_input['attention_mask']  # (batch_size, seq_length)

            # Move inputs to the device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Get token embeddings from the transformer
            outputs = transformer_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            # Last hidden state is the token embeddings
            token_embeddings = outputs.last_hidden_state  # (batch_size, seq_length, embed_dim)

        return token_embeddings, attention_mask
    
    def forward(self, doc_texts, query_texts):
        D_embeddings, D_attention_mask = self.get_token_embeddings(self.doc_transformer, doc_texts, doc=True)
        Q_embeddings, Q_attention_mask = self.get_token_embeddings(self.query_transformer, query_texts)
        
        # Convert attention masks to key_padding_masks
        # In key_padding_mask, True values indicate positions that should be masked
        D_padding_mask = ~D_attention_mask.bool()  # Invert mask
        Q_padding_mask = ~Q_attention_mask.bool()
        
        # Compute the relevance score
        scores = self.interaction_module(
            Q_embeddings, D_embeddings,
            Q_padding_mask=Q_padding_mask,
            D_padding_mask=D_padding_mask
        )
        return scores  # (batch_size)
    
