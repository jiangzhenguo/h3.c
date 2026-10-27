import sys, subprocess
sys.path.insert(0, '/tmp/h3bench')
from sweep import run

# wait for the tile_curve job (F32 phase) to finish
while subprocess.run(["pgrep","-f","tile_curve.py"],capture_output=True).returncode == 0:
    import time; time.sleep(20)

# Paired A/B: each adjacent pair is measured ~1 minute apart, so thermal drift
# cancels in the RATIO, not in the absolute time.
def paired(a, b, rounds, prec="bf16"):
    out = []
    for r in range(rounds):
        ra = run("fork", prec, a, label=f"p{a}-{r}")
        rb = run("fork", prec, b, label=f"p{b}-{r}")
        out.append((a, ra["decode_s"], b, rb["decode_s"], rb["decode_s"]/ra["decode_s"]))
        print(f"PAIR {a}={ra['decode_s']:.2f}s  {b}={rb['decode_s']:.2f}s  ratio {b}/{a} = {rb['decode_s']/ra['decode_s']:.3f}", flush=True)
    return out

r1 = paired(256, 304, 4)
r2 = paired(256, 288, 3)
import statistics
for name, r in (("256 vs 304", r1), ("256 vs 288", r2)):
    ratios = [x[4] for x in r]
    print(f"SUMMARY {name}: ratios {[round(x,3) for x in ratios]} median {statistics.median(ratios):.3f}", flush=True)
print("PAIRED DONE")
