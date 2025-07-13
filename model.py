import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataclasses import dataclass

@dataclass
class Config:
    ctx: int = 1024
    vocab_size: int = 50257 # 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # qkv attention
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # accumulate info from all heads
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x: torch.tensor):
        qkv = self.c_attn(x)
        qkv = qkv.split(self.n_embd, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.n_head), qkv)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.c_proj(y)
        return y

class FeedForwardNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh") # original gpt2 uses approximation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForwardNetwork(config)

    def forward(self, x: torch.tensor):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x
    

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.residual_scale =  1.0 / math.sqrt(2 * config.n_layer) # n_layer is scaled by amount of residual connections
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.ctx, config.n_embd),
                h = nn.ModuleList([TransformerBlock(config) for layer in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # tie weights
        self.apply(self._init_weights)
        self._scale_residual()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and (name.endswith("attn.c_proj") or name.endswith("mlp.c_proj")):
                module.weight.data *= self.residual_scale


    def forward(self, idx: torch.tensor, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss

    def configure_optim(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # separate params by dimensionality
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"decay params: {num_decay_params}, nodecay params: {num_nodecay_params}")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optim = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), fused=use_fused)
        return optim
    

