"""CPU sanity test of the observer on a toy bf16 model: the write model must reproduce the
stored change on nearly every coordinate and the pooled overshoot to a small gap."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train_observe as T
torch.manual_seed(0)
class Toy(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.model = torch.nn.Module()
        s.model.embed_tokens = torch.nn.Embedding(64, 32)
        s.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for L in s.model.layers:
            L.self_attn = torch.nn.Module(); L.mlp = torch.nn.Module()
            L.self_attn.q_proj = torch.nn.Linear(32, 32, bias=False)
            L.mlp.up_proj = torch.nn.Linear(32, 48, bias=False)
            L.input_layernorm = torch.nn.Module(); L.input_layernorm.weight = torch.nn.Parameter(torch.ones(32))
        s.model.norm = torch.nn.Module(); s.model.norm.weight = torch.nn.Parameter(torch.ones(32))
    def forward(s, ids):
        h = s.model.embed_tokens(ids)
        for L in s.model.layers:
            h = h + L.self_attn.q_proj(h) * L.input_layernorm.weight
            h = h + L.mlp.up_proj(h)[..., :32]
        return (h * s.model.norm.weight) @ s.model.embed_tokens.weight.T
m = Toy().to(torch.bfloat16)
opt = torch.optim.AdamW(m.parameters(), lr=2e-4, fused=False)
obs = T.Observer(m, opt)
for step in range(1, 4):
    ids = torch.randint(0, 64, (1, 16))
    loss = torch.nn.functional.cross_entropy(m(ids)[0].float(), ids[0])
    opt.zero_grad(); loss.backward()
    obs.snapshot(); opt.step()
    rec, _ = obs.measure(step)
    w = rec["whole_model_bf16"]
    print(step, {k: round(v, 5) if isinstance(v, float) else v for k, v in w.items() if k not in ("shadow", "x_hist_log2")})
    print("   fam rho:", {f: round(rec["families"][f]["overshoot"], 4) for f in T.FAMILIES})
    print("   shadow:", [round(s["overshoot"], 4) for s in w["shadow"]])
