"""Small JEPA: context-only video encoder, predictor with future queries, EMA target encoder."""
import copy, math
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(s, d, heads):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x):
        h = s.n1(x); x = x + s.attn(h, h, h, need_weights=False)[0]
        return x + s.mlp(s.n2(x))


class Encoder(nn.Module):
    """Patch 'patch'x'patch', temporal patch 1, learned spatial+temporal pos, pre-LN."""
    def __init__(s, res=32, patch=8, d=64, depth=2, max_t=16):
        super().__init__()
        s.patch, s.d, s.np = patch, d, (res // patch) ** 2
        s.embed = nn.Conv2d(3, d, patch, patch)
        s.pos_s = nn.Parameter(torch.randn(1, 1, s.np, d) * 0.02)
        s.pos_t = nn.Parameter(torch.randn(1, max_t, 1, d) * 0.02)
        s.blocks = nn.ModuleList([Block(d, max(1, d // 32)) for _ in range(depth)])
        s.norm = nn.LayerNorm(d)
    def forward(s, x, return_layers=False):               # x [B,T,3,R,R]
        B, T = x.shape[:2]
        t = s.embed(x.flatten(0, 1)).flatten(2).transpose(1, 2).view(B, T, s.np, s.d)
        t = (t + s.pos_s + s.pos_t[:, :T]).reshape(B, T * s.np, s.d)
        pooled = []
        for blk in s.blocks:
            t = blk(t); pooled.append(t.mean(1))
        t = s.norm(t)
        return (t, torch.stack(pooled, 1)) if return_layers else t   # tokens, per-layer pooled [B,L,d]


class JEPA(nn.Module):
    def __init__(s, res=32, patch=8, d=64, depth=2, pred_depth=2, max_t=16, max_off=32):
        super().__init__()
        s.enc = Encoder(res, patch, d, depth, max_t)
        s.tgt = copy.deepcopy(s.enc)
        for p in s.tgt.parameters(): p.requires_grad_(False)
        s.q_pos = nn.Parameter(torch.randn(1, s.enc.np, d) * 0.02)
        s.off_emb = nn.Embedding(max_off + 1, d)
        s.pblocks = nn.ModuleList([Block(d, max(1, d // 32)) for _ in range(pred_depth)])
        s.pnorm, s.pout = nn.LayerNorm(d), nn.Linear(d, d)

    def z(s, ctx):
        """History vector: mean-pooled final context tokens, plus per-layer pooled vectors."""
        tok, layers = s.enc(ctx, return_layers=True)
        return tok.mean(1), layers

    def loss(s, ctx, tgt_frame, off):
        """ctx [B,c,3,R,R]; tgt_frame [B,3,R,R]; off [B] long (future offset in frames)."""
        tok = s.enc(ctx)
        q = s.q_pos + s.off_emb(off)[:, None]
        x = torch.cat([tok, q], 1)
        for b in s.pblocks: x = b(x)
        pred = s.pout(s.pnorm(x[:, tok.shape[1]:]))
        with torch.no_grad():
            tgt = s.tgt(tgt_frame[:, None])                    # [B,np,d]
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
        return F.mse_loss(pred, tgt), pred.detach(), tok.detach()

    @torch.no_grad()
    def ema(s, m):
        for p, q in zip(s.enc.parameters(), s.tgt.parameters()): q.mul_(m).add_(p.detach(), alpha=1 - m)


def effective_rank_fraction(f):
    f = f - f.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(f)
    p = sv / sv.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()) / f.shape[1])
