from src.self_attention import MultiHeadAttention
import torch
import torch.nn as nn
from pydantic import BaseModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # vocab size, same as BPE tokenizer
    "context_length": 1024,  # context length
    "emb_dim": 768,  # embedding dimension
    "n_heads": 12,  # n attention heads
    "n_layers": 12,  # n layers
    "drop_rate": 0.1,  # dropout rate
    "qkv_bias": False,  # query-key-value bias
}


class GptConfig(BaseModel):
    vocab_size: int
    context_length: int
    embedding_dim: int
    n_heads: int
    n_layers: int
    dropout_rate: float
    qkv_bias: bool


class LayerNorm(nn.Module):
    """Normalize input for mean = 0 and variance = 1"""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        # learnable params
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """some dark arts alternative activation to ReLU"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


# feed forward with GELU
class FeedForward(nn.Module):
    def __init__(self, cfg: GptConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim),
            GELU(),
            nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GptConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.embedding_dim,
            d_out=cfg.embedding_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.embedding_dim)
        self.norm2 = LayerNorm(cfg.embedding_dim)
        self.drop_shortcut = nn.Dropout(cfg.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # add shortcut and layer normalisation to attention
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        # add shortcut and layer normalisation to feed forward
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GptConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.drop_emb = nn.Dropout(cfg.dropout_rate)
        # n_layers transformer blocks, each with self attention heads
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        # normalise final output
        self.final_norm = LayerNorm(cfg.embedding_dim)
        # project output to same dimensions as input
        self.out_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # device determined at this stage
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # embedding dim x vocab size
        logits = self.out_head(x)
        return logits


def generate_text_simple(
    model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Simple func for greedy generation"""
    for _ in range(max_new_tokens):
        # index is (batch, n_tokens) , with n_tokens being a vector of token indexes

        # truncated to context size
        idx_truncated = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_truncated)

        # Focuses only on the last time step, so that (batch, n_token, vocab_size)
        # becomes (batch, vocab_size)
        # this is needed to generate distribution over vocab
        logits = logits[:, -1, :]
        # has shape (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        # has shape (batch, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # append sample index to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    # return combined token indexes
    return idx
