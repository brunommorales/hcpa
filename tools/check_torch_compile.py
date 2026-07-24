#!/usr/bin/env python3
"""Prova (ou refuta) que o `torch.compile` do pytorch_opt está de fato compilando.

Por que isto existe
-------------------
`pytorch_opt/dr_hcpa_v2_2024.py` faz:

    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, mode="reduce-overhead")

Com `suppress_errors=True`, QUALQUER falha do backend cai em **eager, em silêncio**.
O log não mostra erro nenhum. Se isso acontecer, a comparação central do estudo
deixa de ser "XLA vs TorchInductor" e passa a ser "XLA vs eager" — sem que nada
no CSV denuncie.

Contar kernels NÃO resolve: `mode="reduce-overhead"` usa CUDA graphs, que reduz o
overhead de *lançamento*, não o número de kernels. O CUPTI vê os mesmos kernels.

O que este script faz
---------------------
Reproduz o step de treino real (autocast + GradScaler + clip + optimizer.step) e
mede quatro coisas que só são compatíveis com "compilou de verdade":

  1. `torch._dynamo.utils.counters` -> frames compilados > 0 e graph breaks
  2. `torch._inductor` gerou kernels Triton? (conta arquivos no cache)
  3. CUDA graphs realmente capturados? (contador do cudagraph_trees)
  4. tempo/step eager vs compiled, em regime estável

Uso (dentro do container pytorch_opt, com GPU):
    python3 tools/check_torch_compile.py --model inception_v3 --img 299 --batch 96
"""
import argparse
import os
import time

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/hcpa_inductor_cache")


def build(model_name, img):
    import timm
    import torch
    m = timm.create_model(model_name, pretrained=False, num_classes=1)
    if hasattr(m, "aux_logits"):
        m.aux_logits = False
    return m


def bench(model, x, y, use_amp, steps, warmup, label):
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lossf = torch.nn.BCEWithLogitsLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x).squeeze(1)
            loss = lossf(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(f"  [{label}] {dt*1000:.1f} ms/step   ({steps} steps, {warmup} warmup)")
    return dt


def count_triton_kernels():
    """Kernels Triton gerados pelo TorchInductor ficam em cache como .py/.cubin."""
    root = os.environ["TORCHINDUCTOR_CACHE_DIR"]
    n = 0
    for dirpath, _, files in os.walk(root):
        n += sum(1 for f in files if f.endswith((".py", ".cubin", ".so")))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inception_v3")
    ap.add_argument("--img", type=int, default=299)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    import torch
    print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    if not torch.cuda.is_available():
        raise SystemExit("precisa de GPU")

    dev = torch.device("cuda")
    x = torch.randn(a.batch, 3, a.img, a.img, device=dev)
    y = (torch.rand(a.batch, device=dev) > 0.5).float()

    # ---------------- EAGER ----------------
    print("\n=== 1) EAGER (referência) ===")
    m_eager = build(a.model, a.img).to(dev)
    t_eager = bench(m_eager, x, y, True, a.steps, a.warmup, "eager")
    del m_eager
    torch.cuda.empty_cache()

    # ---------------- COMPILED (exatamente como o pytorch_opt faz) ----------------
    print("\n=== 2) COMPILED (mesma config do pytorch_opt) ===")
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.utils.counters.clear()
    n_before = count_triton_kernels()

    dynamo.config.suppress_errors = True          # <-- exatamente como no código real
    m = build(a.model, a.img)
    m = torch.compile(m, mode="reduce-overhead")  # <-- exatamente como no código real
    m = m.to(dev)

    t_comp = bench(m, x, y, True, a.steps, a.warmup, "compiled")

    # ---------------- VEREDITO ----------------
    print("\n=== 3) EVIDÊNCIA ===")
    c = dynamo.utils.counters
    frames_ok = c["frames"]["ok"]
    frames_tot = c["frames"]["total"]
    breaks = sum(c["graph_break"].values())
    n_after = count_triton_kernels()
    triton = n_after - n_before

    print(f"  dynamo frames compilados : {frames_ok} / {frames_tot}")
    print(f"  graph breaks             : {breaks}")
    print(f"  kernels Triton gerados   : {triton}")
    if breaks:
        top = sorted(c["graph_break"].items(), key=lambda kv: -kv[1])[:3]
        for k, v in top:
            print(f"      break x{v}: {str(k)[:80]}")

    # cudagraphs realmente capturados?
    cg = "?"
    try:
        from torch._inductor.cudagraph_trees import get_manager
        mgr = get_manager(0, create_if_none_exists=False)
        cg = "sim" if (mgr is not None and getattr(mgr, "roots", None)) else "nao"
    except Exception as e:
        cg = f"nao inspecionavel ({type(e).__name__})"
    print(f"  CUDA graphs capturados   : {cg}")

    print(f"\n  eager    : {t_eager*1000:.1f} ms/step")
    print(f"  compiled : {t_comp*1000:.1f} ms/step")
    speed = t_eager / t_comp
    print(f"  speedup  : {speed:.2f}x")

    print("\n=== VEREDITO ===")
    compilou = frames_ok > 0 and triton > 0
    if not compilou:
        print("  ❌ NAO COMPILOU. O treino roda em EAGER.")
        print("     A comparacao do estudo e 'XLA vs eager', nao 'XLA vs TorchInductor'.")
        return 1
    if speed < 1.05:
        print(f"  ⚠️  Compilou ({frames_ok} frames, {triton} kernels Triton), mas o ganho")
        print(f"     e desprezivel ({speed:.2f}x). Provavel graph break dominante ou")
        print("     cudagraphs desativado. Verificar os breaks acima.")
        return 2
    print(f"  ✅ COMPILOU: {frames_ok} frames, {triton} kernels Triton, {speed:.2f}x mais rapido.")
    print("     A rotulagem 'torch.compile / TorchInductor' e legitima.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
