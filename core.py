import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# import tiktoken 

class DataManager:
    def __init__(self, data_source_path: str = None, batch_size: int = 32, block_size: int = 8):
        self.text = None
        self.batch_size = batch_size
        self.block_size = block_size

        if not os.path.exists(data_source_path):
            raise FileNotFoundError(f"The file '{data_source_path}' does not exist.")
        if not os.path.basename(data_source_path).endswith(".txt"):
            raise ValueError("Data source must be a text file with a .txt extension.")
        
        with open(data_source_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.define_custom_encoder()
        self.encoded_mat = torch.tensor(self.encoder(self.text), dtype=torch.long)
        self.train_test_split()

    @property
    def vocab_size(self):
        return len(list(set(self.text)))

    def train_test_split(self, train_size: float = 0.85):
        n = int(train_size * len(self.encoded_mat))
        self.train_data = self.encoded_mat[:n]
        self.test_data = self.encoded_mat[n:]

    def get_batch(self, split_type: str = "train", device: str = "cpu"):
        data = self.train_data if split_type == 'train' else self.test_data
        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))
        X = torch.stack([data[i:i + self.block_size] for i in idx])
        Y = torch.stack([data[i + 1:i + self.block_size + 1] for i in idx])
        X, Y = X.to(device), Y.to(device)
        return X, Y

    def define_custom_encoder(self):
        all_tokens = sorted(list(set(self.text)))
        string_to_num_mapper = {t: i for i, t in enumerate(all_tokens)}
        num_to_string_mapper = {i: t for i, t in enumerate(all_tokens)}
        self.encoder = lambda txt: [string_to_num_mapper[s] for s in txt]
        self.decoder = lambda enc: "".join([num_to_string_mapper[e] for e in enc])

####################################################################################################################

class Head(nn.Module):
    """ Single self attention head """
    def __init__(self, embedding_dim: int, head_size: int, dropout: float = 0.3):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # scaled dot product attention from paper
        weights = (query @ key.transpose(-2, -1)) * (self.head_size ** -0.5)
        
        # masking future tokens 
        tril = torch.tril(torch.ones(T, T, device=x.device))
        weights = weights.masked_fill(tril == 0, float('-inf'))
        weights = self.dropout(F.softmax(weights, dim=-1))
        out = weights @ value
        return out

class MultiHeadAttention(nn.Module):
    """ Multi self attention heads """
    def __init__(self, no_of_heads, head_size, embedding_dim, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size, embedding_dim=embedding_dim, dropout=dropout) for _ in range(no_of_heads)])
        self.ln = nn.Linear(head_size * no_of_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.ln(out))
        return out

class FeedFoward(nn.Module):
    """ A simple linear layer followed by a GELU """
    def __init__(self, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embedding_dim: int, no_of_heads: int, dropout: float = 0.3):
        super().__init__()
        head_size = embedding_dim // no_of_heads
        self.multi_head_attn = MultiHeadAttention(no_of_heads=no_of_heads, head_size=head_size, embedding_dim=embedding_dim, dropout=dropout)
        self.ff_net = FeedFoward(embedding_dim=embedding_dim, dropout=dropout)
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.ln_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.multi_head_attn(self.ln_1(x))
        x = x + self.ff_net(self.ln_2(x))
        return x


####################################################################################################################


class charGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embedding_dim: int, no_of_attn_layers: int = 4, no_of_heads: int = 6, dropout: float = 0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, no_of_heads, dropout) for _ in range(no_of_attn_layers)])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.ff_final = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        device = input_ids.device
        token_embedding = self.token_embedding_table(input_ids)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.ff_final(x)

        if targets is None:
            loss = None
        else:
            logits_reshaped = logits.view(B * T, -1)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    def generate(self, idx, max_new_tokens, device):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:].to(device)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    
        


    



    


    







    
            
    
