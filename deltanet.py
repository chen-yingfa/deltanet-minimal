from dataclasses import dataclass
import json
from typing import Optional

from safetensors.torch import load_file
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange


@dataclass
class DeltaNetConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    initializer_range: float = 0.02
    hidden_ratio: float = 4.0
    expand_k: float = 1.0
    expand_v: float = 1.0
    num_heads: int = 8
    num_hidden_layers: int = 24
    tie_word_embeddings: bool = False
    conv_size: int = 4
    norm_eps: float = 1e-6
    pad_token_id: int = 2
    eos_token_id: int = 2
    bos_token_id: int = 1
    chunk_size: int = 32

    @classmethod
    def from_pretrained(cls, path):
        obj = cls()
        config = json.load(open(path, "r"))
        for attr in obj.__dict__:
            if attr in config:
                setattr(obj, attr, config[attr])
        return obj


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return x.to(dtype=orig_dtype)


def delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, nheads, seqlen, dk = q.shape
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)  # (B, nheads, len, dv)
    S = torch.zeros(b, nheads, dk, d_v).to(v)
    q = q * (dk**-0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]  # (B, nheads, len, 1)

    if initial_state is not None:
        S += initial_state

    for i in range(seqlen):
        _k = k[:, :, i]  # (B, nheads, dk)
        _q = q[:, :, i]  # (B, nheads, dk)
        _v = v[:, :, i].clone()  # (B, nheads, dv)
        beta_i = beta[:, :, i]  # (B, nheads, 1)
        _v = _v - torch.einsum("bhdm,bhd->bhm", S.clone(), _k)  # (B, nheads, dv)
        _v = _v * beta_i  # (B, nheads, dv)
        S = S.clone() + torch.einsum("bhd,bhm->bhdm", _k, _v)  # (B, nheads, dk, dv)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def delta_rule_chunkwise(
    q, k, v, beta, initial_state: None | Tensor = None, chunk_size=32
):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k**-0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    assert l % chunk_size == 0

    # compute (I - tri(diag(beta) KK^T))^{-1}
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0
    )
    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        [q, k, v, k_beta],
    )
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i] + (
            attn[..., i, :, None].clone() * attn[..., :, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    u = attn @ v
    w = attn @ k_beta
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1
    )
    for i in range(0, l // chunk_size):
        q_i, k_i = q[:, :, i], k[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2)).masked_fill_(mask, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S
        o_inter = q_i @ S
        o[:, :, i] = o_inter + attn @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i

    return rearrange(o, "b h n c d -> b h (n c) d"), S


def delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    b, h, l, d_k = q.shape
    # d_v = v.shape[-1]
    q = q * (d_k**-0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=BN), [q, k, v, k_beta]
    )
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (
            T[..., i, :, None].clone() * T[..., :, :i].clone()
        ).sum(-2)
    T = T + torch.eye(BN, dtype=torch.float, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = (
        k
        - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    )
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(
        lambda x: rearrange(x, "b h n c d -> b h (n c) d"), [q, k, v, k_beta, o_intra]
    )
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i : i + BM]
        o_i = o_intra[:, :, i : i + BM]
        # intra block
        for j in range(i + BM - 2 * BN, i - BN, -BN):
            k_j = k[:, :, j : j + BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i + BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i : i + BM, j : j + BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j : j + BN]
            o_i += A_ij @ v[:, :, j : j + BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j : j + BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i : i + BM, j : j + BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j : j + BN]
            o_i += A_ij @ v[:, :, j : j + BN]
        o[:, :, i : i + BM] = o_i

    for i in range(0, l // BN):
        A[:, :, i * BN : i * BN + BN, i * BN : i * BN + BN] = A_local[:, :, i]

    return o, A


class MLP(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()
        intermediate_size = int(config.hidden_size * config.hidden_ratio * 2 / 3)
        intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.up_proj(x) * self.act(self.gate_proj(x))
        return self.down_proj(x)


class DeltaNet(nn.Module):
    def __init__(self, config: DeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.conv_size = config.conv_size

        self.k_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.conv_size,
            groups=config.hidden_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.conv_size,
            groups=config.hidden_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.q_conv1d = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.conv_size,
            groups=config.hidden_size,
            padding=self.conv_size - 1,
            bias=False,
        )
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.b_proj = nn.Linear(config.hidden_size, config.num_heads, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def short_conv(
        self,
        x: Tensor,
        conv_module: nn.Conv1d,
        state: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        orig_len = x.shape[1]
        if state is not None:
            x = torch.cat((state, x), dim=1)

        # The new short conv state
        new_state = x[:, -self.conv_size + 1 :]  # (B, conv_size - 1, D)

        x = rearrange(x, "b t d -> b d t")
        x = conv_module(x)
        x = rearrange(x, "b d t -> b t d")  # (B, len, D)
        # Remove padding on the right
        x = x[:, : -(self.conv_size - 1)]

        # Remove padding on the left
        x = x[:, -orig_len:]  # (B, len, D)

        return x, new_state

    def forward(self, x: Tensor, state: Optional[tuple[tuple, Tensor]] = None):
        # print(f"DeltaNet.forward {self.layer_idx}", x.shape, x)
        batch_size, seqlen, _ = x.shape
        q = self.q_proj(x)  # (B, len, D)
        k = self.k_proj(x)  # (B, len, D)
        v = self.v_proj(x)  # (B, len, D)
        beta = self.b_proj(x).sigmoid()  # (B, len, nheads)

        # ==== short conv ====
        if state is None:
            q_conv_state, k_conv_state, v_conv_state = None, None, None
        else:
            q_conv_state, k_conv_state, v_conv_state = state[0]
        q, q_conv_state = self.short_conv(
            q, state=q_conv_state, conv_module=self.q_conv1d
        )
        k, k_conv_state = self.short_conv(
            k, state=k_conv_state, conv_module=self.k_conv1d
        )
        v, v_conv_state = self.short_conv(
            v, state=v_conv_state, conv_module=self.v_conv1d
        )
        new_conv_state = (q_conv_state, k_conv_state, v_conv_state)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_heads)
        beta = rearrange(beta, "b l h -> b h l")

        # ==== QK norm ====
        q = F.normalize(q, dim=-1, p=2, eps=self.config.norm_eps)
        k = F.normalize(k, dim=-1, p=2, eps=self.config.norm_eps)

        # ==== delta rule ====
        recurrent_state = state[1] if state is not None else None
        # out: [B, H, len, D]
        out, new_recurrent_state = delta_rule_recurrence(
            q, k, v, beta=beta, initial_state=recurrent_state
        )

        # ==== Output projection ====
        out = self.o_norm(out)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.o_proj(out)

        new_state = (new_conv_state, new_recurrent_state)
        return out, new_state


class DeltaNetLayer(nn.Module):
    def __init__(self, config: DeltaNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = DeltaNet(config=config, layer_idx=layer_idx)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, state: tuple | None = None):
        """
        x: (B, len, D)
        state: (conv_state, recurrent_state)
        """
        # ==== Token mixing ====
        res = x
        x = self.attn_norm(x)
        x, state = self.attn(x, state)
        x += res

        # ==== MLP ====
        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x) + res
        return x, state


class DeltaNetModel(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [DeltaNetLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, input_ids: Tensor, state=None):
        x = self.embeddings(input_ids)

        if state is None:
            state = [None for _ in range(self.config.num_hidden_layers)]

        for i, layer in enumerate(self.layers):
            x, state[i] = layer(x, state[i])

        x = self.norm(x)
        return x, state


class DeltaNetForCausalLM(nn.Module):
    def __init__(self, config: DeltaNetConfig):
        super().__init__()
        self.model = DeltaNetModel(config=config)
        self.chunk_size = 32
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, state=None):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        assert input_ids.ndim == 2, f"Got input_ids with shape {input_ids.shape}"

        x, state = self.model(input_ids, state)

        logits = self.lm_head(x)  # (B, len, vocab_size)
        return logits, state

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        eos_token_id: int = 0,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        bsz = input_ids.shape[0]
        last_token = input_ids
        state = None

        # Generate
        output_ids = [input_ids]
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out, state = self.forward(last_token, state)
            logits = out[:, -1]  # (B, V)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                # Remove all tokens with a probability less than the top_k token
                topk_logits = torch.topk(logits, k=top_k, dim=-1)[0]  # (B)
                # print(topk_logits.shape, topk_logits)
                # exit()
                for bi in range(bsz):
                    indices_to_remove = logits[bi] < topk_logits[bi, -1]
                    logits[bi, indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > 0.5
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)  # (B, V)
            last_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            output_ids.append(last_token)
        output_ids = torch.cat(output_ids, dim=-1)  # (B, 1)
        return output_ids


if __name__ == "__main__":
    from transformers import AutoTokenizer

    config_path = "configs/1.3b.json"
    ckpt_path = "model.safetensors"
    tok_path = "fla-hub/delta_net-1.3B-100B"
    device = "mps"

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    # tokenizer.save_pretrained('tokenizer')
    config = DeltaNetConfig.from_pretrained(config_path)
    print(config)
    model = DeltaNetForCausalLM(config=config).to(device, dtype=torch.bfloat16)
    model.eval()
    print(model)
    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)

    prompt = "My name is"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print("====== prompt ======")
    print(prompt)
    print("====================")
    outputs = model.generate(input_ids, max_new_tokens=20)
    output_text = tokenizer.batch_decode(outputs)
    print("====== output ======")
    print(output_text)
    print("====================")
    # This should output:
    # ['<s> My name is Katie and I am a 20 year old student at the University of North Carolina at Chap']
