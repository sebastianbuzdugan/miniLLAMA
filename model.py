import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import logging
import os
from utils import create_masks


# =========================
# Logging Configuration
# =========================

# Create a logger for the model
logger = logging.getLogger('model_logger')
logger.setLevel(logging.INFO)  # Set to INFO or DEBUG based on your needs

# Create a file handler to write logs to 'model_log.txt'
log_file = os.path.join(os.path.dirname(__file__), "model_log.txt")
file_handler = logging.FileHandler(log_file, mode='w')  # 'w' to overwrite each run
file_handler.setLevel(logging.INFO)  # Adjust level as needed

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Optionally, add a stream handler to also output to console (set to WARNING to reduce verbosity)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)  # Only WARNING and above will be printed to console
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# =========================
# Model Components
# =========================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        https://arxiv.org/pdf/1910.07467.pdf
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(hidden_states.dtype)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.intermediate_size = int(
            4 * self.config["hidden_size"] * (2 / 3))  # https://arxiv.org/pdf/2302.13971.pdf PAGE 3
        self.c_fc = nn.Linear(self.config["hidden_size"], self.intermediate_size, bias=False)
        self.v_proj = nn.Linear(self.config["hidden_size"], self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.config["hidden_size"], bias=False)

    def forward(self, x):
        x = F.silu(self.c_fc(x)) * self.v_proj(x)
        return self.c_proj(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids, seq_len=None):
        # x: [B, num_key_value_heads, T, head_dim]
        # position_ids: [1, T]
        freqs = (self.inv_freq[:, None].float() * position_ids.float()).transpose(0, 1)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        return emb.cos(), emb.sin()


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Applies rotary positional embeddings to query and key tensors.
    Args:
        q: Query tensor [B, num_heads, T, head_dim]
        k: Key tensor [B, num_heads, T, head_dim]
        cos: Cosine embeddings [T, head_dim]
        sin: Sine embeddings [T, head_dim]
        position_ids: Position IDs [1, T]
        unsqueeze_dim: Dimension to unsqueeze for broadcasting
    Returns:
        q_embed, k_embed: Embedding-applied query and key tensors
    """
    # Expand cos and sin to [1, 1, T, head_dim] for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.n_rep = self.config["n_local_heads"] // self.config["n_local_kv_heads"]
        self.num_heads = config["n_head"]
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = self.config["attention_dropout"]

        # Validate that num_heads is divisible by num_key_value_heads
        if self.num_key_value_groups < 1:
            logger.error("num_key_value_heads must divide num_heads without remainder.")
            raise ValueError("num_key_value_heads must divide num_heads without remainder.")

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config["max_len"],
            base=config["rope_theta"],
            device=None  # Device will be set dynamically based on input
        )

    def forward(self, x, mask=None, training=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Generate position_ids as [1, T] instead of [B, T]
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).long()  # [1, T]

        q = self.wq(x)  # [B, T, num_heads * head_dim]
        k = self.wk(x)  # [B, T, num_key_value_heads * head_dim]
        v = self.wv(x)  # [B, T, num_key_value_heads * head_dim]

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, num_key_value_heads, T, head_dim]
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [B, num_key_value_heads, T, head_dim]

        # Change 1: Log shapes of q, k, v
        logger.info(f"Change 1: Shape of q: {q.shape}, Shape of k: {k.shape}, Shape of v: {v.shape}")

        # Check for zero-size dimension in k or v to prevent errors in matmul
        if k.size(1) == 0 or v.size(1) == 0:
            logger.warning("Change 1: Skipping attention computation due to zero dimension in k or v")
            return torch.zeros_like(q)  # Return zeros or handle as appropriate for your model

        # Apply rotary positional embeddings
        cos, sin = self.rotary_emb(v, position_ids, seq_len=None)  # [T, head_dim], [T, head_dim]
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        k = repeat_kv(k, self.num_key_value_groups)  # [B, num_heads, T, head_dim]
        v = repeat_kv(v, self.num_key_value_groups)  # [B, num_heads, T, head_dim]

        matmul_qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, T, T]
        if mask is not None:
            matmul_qk += (mask * -1e9)

        attn_scores = F.softmax(matmul_qk, dim=-1)  # [B, num_heads, T, T]
        attn_scores = F.dropout(attn_scores, p=self.attention_dropout, training=self.training)
        y = torch.matmul(attn_scores, v)  # [B, num_heads, T, head_dim]

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, hidden_size]

        return self.c_proj(y)


class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config["hidden_size"], eps=config["eps"])
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config["hidden_size"], eps=config["eps"])
        self.mlp = FeedForward(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class LLAMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config["vocab_size"] is not None
        assert config["block_size"] is not None
        self.config = config

        self.vocab_size = config["vocab_size"]

        self.transformer = nn.ModuleDict(dict(
            embedding_layer=nn.Embedding(self.vocab_size, config["hidden_size"]),
            h=nn.ModuleList([DecoderBlock(config) for _ in range(config["n_layer"])]),
            layer_norm=RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"]),
        ))
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.transformer.embedding_layer.weight = self.lm_head.weight
        logger.info("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        taken this method from Andrej Karpathy's minGPT
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.embedding_layer.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        mask = create_masks(idx, device)  # Creating mask to handle left to right attention and mask
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        x = self.transformer.embedding_layer(idx)  # token embeddings of shape (b, t, embd)

        for decoder_block in self.transformer.h:
            x = decoder_block(x, mask)
        x = self.transformer.layer_norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, prompt, tokenizer, max_new_tokens=64, temperature=1.0, top_k=None):
        # [BOS] index is 3 and [EOS] index is 4

        input_ids = torch.tensor([[3] + tokenizer.EncodeAsIds(prompt)], device=next(self.parameters()).device)
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, predicted_id), dim=1)

            if predicted_id.item() == 4:
                break

        input_ids = input_ids[0]
        return tokenizer.decode_ids(input_ids.cpu().numpy().tolist()[1:])
