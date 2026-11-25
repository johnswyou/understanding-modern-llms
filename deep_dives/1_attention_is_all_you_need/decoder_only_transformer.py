import math
import torch

torch.manual_seed(0)
text = open("papers/twelve-essential/markdown_output/1706.03762-attention-is-all-you-need.md").read()
chars = sorted(set(text))                      # the vocabulary: every distinct character
stoi = {c: i for i, c in enumerate(chars)}     # character -> token id
data = torch.tensor([stoi[c] for c in text])   # (len(text),)  the whole text as token ids
n_vocab = len(chars)                           # 105 for this text

B, n_ctx = 32, 64
d_model, h, d_ff, N = 64, 4, 256, 2
d_k = d_model // h                             # 16
p_drop = 0.1

def weight(fan_in, fan_out):                   # (fan_in, fan_out), entries with std 1/sqrt(fan_in)
    return (torch.randn(fan_in, fan_out) / math.sqrt(fan_in)).requires_grad_()

def zeros(size): return torch.zeros(size, requires_grad=True)
def ones(size):  return torch.ones(size, requires_grad=True)

params = {
    "E": (torch.randn(n_vocab, d_model) / math.sqrt(d_model)).requires_grad_(),  # (n_vocab, d_model)
    "blocks": [],
}
for _ in range(N):
    params["blocks"].append({
        "W_q": weight(d_model, d_model),       # (d_model, h*d_k): the h heads' query matrices side by side
        "W_k": weight(d_model, d_model),       # (d_model, h*d_k)
        "W_v": weight(d_model, d_model),       # (d_model, h*d_k)
        "W_o": weight(d_model, d_model),       # (h*d_k, d_model): mixes the concatenated heads
        "ln1_g": ones(d_model),  "ln1_b": zeros(d_model),   # LayerNorm after attention: gamma, beta
        "W_1": weight(d_model, d_ff), "b_1": zeros(d_ff),   # FFN layer 1: (d_model, d_ff), (d_ff,)
        "W_2": weight(d_ff, d_model), "b_2": zeros(d_model),# FFN layer 2: (d_ff, d_model), (d_model,)
        "ln2_g": ones(d_model),  "ln2_b": zeros(d_model),   # LayerNorm after FFN: gamma, beta
    })

def sinusoidal_pe(n_ctx, d_model):
    pos = torch.arange(n_ctx).float()[:, None]               # (n_ctx, 1)          0, 1, ..., n_ctx-1
    two_i = torch.arange(0, d_model, 2).float()[None, :]     # (1, d_model/2)      0, 2, 4, ..., d_model-2
    angle = pos / 10000 ** (two_i / d_model)                 # (n_ctx, d_model/2)  pos * omega_i
    pe = torch.zeros(n_ctx, d_model)                         # (n_ctx, d_model)
    pe[:, 0::2] = torch.sin(angle)                           # even dimensions
    pe[:, 1::2] = torch.cos(angle)                           # odd dimensions
    return pe

PE = sinusoidal_pe(n_ctx, d_model)                           # (n_ctx, d_model), not trained

def attention(q, k, v, mask):
    # q, k, v: (B, h, n, d_k)    mask: (n, n) bool, True where key j <= query i
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])    # (B, h, n, n)   S
    scores = scores.masked_fill(~mask, float("-inf"))            # (B, h, n, n)   S + M
    weights = torch.softmax(scores, dim=-1)                      # (B, h, n, n)   A, each row sums to 1
    return weights @ v                                           # (B, h, n, d_k)

def multi_head_attention(x, blk, mask):
    B, n, _ = x.shape                                        # x: (B, n, d_model)
    def split_heads(t):                                      # (B, n, d_model) -> (B, h, n, d_k)
        return t.view(B, n, h, d_k).transpose(1, 2)
    q = split_heads(x @ blk["W_q"])                          # (B, h, n, d_k)
    k = split_heads(x @ blk["W_k"])                          # (B, h, n, d_k)
    v = split_heads(x @ blk["W_v"])                          # (B, h, n, d_k)
    heads = attention(q, k, v, mask)                         # (B, h, n, d_k)
    concat = heads.transpose(1, 2).reshape(B, n, d_model)    # (B, n, h*d_k) = (B, n, d_model)
    return concat @ blk["W_o"]                               # (B, n, d_model)

def ffn(x, blk):
    hidden = torch.relu(x @ blk["W_1"] + blk["b_1"])         # (B, n, d_ff)
    return hidden @ blk["W_2"] + blk["b_2"]                  # (B, n, d_model)

def layer_norm(x, g, b, eps=1e-6):
    mean = x.mean(-1, keepdim=True)                          # (B, n, 1)  mu for each position
    var = ((x - mean) ** 2).mean(-1, keepdim=True)           # (B, n, 1)  sigma^2 for each position
    return g * (x - mean) / torch.sqrt(var + eps) + b        # (B, n, d_model)


def dropout(x, train):
    if not train:
        return x
    keep = (torch.rand_like(x) > p_drop).float()             # same shape as x: 1 = keep, 0 = drop
    return x * keep / (1 - p_drop)


def decoder_block(x, blk, mask, train):
    a = multi_head_attention(x, blk, mask)                                   # (B, n, d_model)
    x = layer_norm(x + dropout(a, train), blk["ln1_g"], blk["ln1_b"])         # (B, n, d_model)
    f = ffn(x, blk)                                                          # (B, n, d_model)
    x = layer_norm(x + dropout(f, train), blk["ln2_g"], blk["ln2_b"])         # (B, n, d_model)
    return x

def forward(params, ids, train=False):
    n = ids.shape[1]                                         # ids: (B, n) token ids
    x = params["E"][ids] * math.sqrt(d_model)                # (B, n, d_model)  embedding lookup
    x = dropout(x + PE[:n], train)                           # (B, n, d_model)  add position
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool))    # (n, n)  True on and below the diagonal
    for blk in params["blocks"]:
        x = decoder_block(x, blk, mask, train)               # (B, n, d_model)
    return x @ params["E"].T                                 # (B, n, n_vocab)  logits

ids = torch.randint(0, n_vocab, (1, 16))                     # (1, 16) a random sequence
ids_changed = ids.clone()
ids_changed[0, 10] = (ids[0, 10] + 1) % n_vocab
out, out_changed = forward(params, ids), forward(params, ids_changed)   # each (1, 16, n_vocab)
assert torch.allclose(out[0, :10], out_changed[0, :10])      # earlier positions cannot see index 10
assert not torch.allclose(out[0, 10:], out_changed[0, 10:])  # index 10 and later can

def loss_fn(logits, targets, eps_ls=0.1):
    # logits: (B, n, n_vocab)    targets: (B, n) token ids
    logp = torch.log_softmax(logits, dim=-1)                 # (B, n, n_vocab)  log p for every token
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)   # (B, n)  -log p(correct token)
    smooth = -logp.mean(-1)                                  # (B, n)  mean of -log p over the vocabulary
    return ((1 - eps_ls) * nll + eps_ls * smooth).mean()     # scalar


def lrate(step, warmup):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


all_tensors = [params["E"]] + [t for blk in params["blocks"] for t in blk.values()]
opt = torch.optim.Adam(all_tensors, betas=(0.9, 0.98), eps=1e-9)
steps, warmup = 2000, 200

for step in range(1, steps + 1):
    for group in opt.param_groups:
        group["lr"] = lrate(step, warmup)
    start = torch.randint(0, len(data) - n_ctx - 1, (B,))             # (B,) random window starts
    x = torch.stack([data[s : s + n_ctx] for s in start])             # (B, n_ctx) inputs
    y = torch.stack([data[s + 1 : s + n_ctx + 1] for s in start])     # (B, n_ctx) targets, shifted by one
    loss = loss_fn(forward(params, x, train=True), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step == 1 or step % 500 == 0:
        with torch.no_grad():                                # plain cross-entropy, no dropout
            ce = loss_fn(forward(params, x), y, eps_ls=0.0).item()
        print(f"step {step:5d}   cross-entropy {ce:.2f}")

@torch.no_grad()
def generate(params, ids, new_tokens):
    for _ in range(new_tokens):                              # ids: (1, n) the text so far
        logits = forward(params, ids[:, -n_ctx:])            # (1, n, n_vocab), at most n_ctx positions
        probs = torch.softmax(logits[:, -1], dim=-1)         # (1, n_vocab)  distribution for the next token
        next_id = torch.multinomial(probs, 1)                # (1, 1)  draw one token
        ids = torch.cat([ids, next_id], dim=1)               # (1, n+1)
    return ids

prompt = torch.tensor([[stoi[c] for c in "The Transformer "]])       # (1, 16)
print("".join(chars[i] for i in generate(params, prompt, 300)[0].tolist()))

n_params = sum(t.numel() for t in all_tensors)
expected = n_vocab * d_model + N * (4 * d_model**2 + 2 * d_model * d_ff + d_ff + d_model + 4 * d_model)
assert n_params == expected
print(f"{n_params:,} parameters")
