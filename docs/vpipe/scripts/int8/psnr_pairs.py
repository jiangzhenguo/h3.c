import numpy as np, os
OUT="/tmp/h3bench/rgb"
def cmp(a,b):
    ma=np.memmap(f"{OUT}/{a}.rgbf32",dtype=np.float32,mode="r")
    mb=np.memmap(f"{OUT}/{b}.rgbf32",dtype=np.float32,mode="r")
    n=ma.size; mx=0.0; sq=0.0
    for i in range(0,n,20_000_000):
        x=np.asarray(ma[i:i+20_000_000],dtype=np.float64)
        y=np.asarray(mb[i:i+20_000_000],dtype=np.float64)
        d=x-y; mx=max(mx,float(np.abs(d).max())); sq+=float((d*d).sum())
    mse=sq/n
    return mx,(mse**0.5),(float('inf') if mse==0 else 10*np.log10(1/mse))
pairs=[
 ("fork-bf16-t256","fork-bf16-t304","同一算法，只改 tile 256->304"),
 ("fork-bf16-t304","73-bf16-t304","同一 tile 304，#80 vs #73 的 BF16"),
 ("73-int8-t256","73-int8-t304","int8，只改 tile"),
 ("fork-bf16-t304","73-int8-t304","tile 304，#80-bf16 vs #73-int8"),
 ("73-bf16-t304","73-f32-t256","#73-bf16(304) vs F32(256)"),
]
for a,b,note in pairs:
    if not (os.path.exists(f"{OUT}/{a}.rgbf32") and os.path.exists(f"{OUT}/{b}.rgbf32")):
        print(f"skip {a} vs {b}"); continue
    mx,rmse,ps=cmp(a,b)
    print(f"{a:<18} vs {b:<18} max_abs={mx:.5f} rmse={rmse:.6f} psnr={ps:6.2f} dB   ({note})")
