"""E3: what does it cost to move KV in scattered blocks instead of contiguously?

Priority transfer requires gathering an arbitrary subset of blocks and sending
them as many small messages. Three costs are measured / modelled:
  (a) host-side gather (memcpy from scattered offsets into a staging buffer)
  (b) device-side gather (torch index_select on the accelerator)
  (c) per-message wire overhead (TCP loopback here; RDMA modelled with
      published constants, since no RDMA NIC is present on this host)
"""
import json, os, socket, struct, threading, time
import numpy as np

def timeit(fn, warmup=2, reps=5):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

# ---------------------------------------------------------------- (a) host gather
def host_gather(total_mb=256):
    src = np.empty(total_mb * 1024 * 1024, dtype=np.uint8); src[:] = 7
    out = {}
    for blk_kb in [4, 16, 64, 256, 1024, 4096]:
        blk = blk_kb * 1024
        n = len(src) // blk
        dst = np.empty(n * blk, dtype=np.uint8)
        idx = np.random.permutation(n)
        def scatter():
            for i, b in enumerate(idx):
                dst[i*blk:(i+1)*blk] = src[b*blk:(b+1)*blk]
        def contig():
            dst[:n*blk] = src[:n*blk]
        ts, tc = timeit(scatter), timeit(contig)
        gb = n * blk / 1e9
        out[blk_kb] = dict(block_kb=blk_kb, n_blocks=n,
                           scatter_gbps=gb/ts, contig_gbps=gb/tc,
                           efficiency=tc/ts)
        print(f"  host gather blk={blk_kb:>5}KB  scatter {gb/ts:7.2f} GB/s  "
              f"contig {gb/tc:7.2f} GB/s  eff {tc/ts:.2f}", flush=True)
    return out

# ---------------------------------------------------------------- (b) device gather
def device_gather(total_mb=256):
    import torch
    if not torch.backends.mps.is_available(): return {}
    dev = "mps"
    out = {}
    for blk_kb in [4, 16, 64, 256, 1024]:
        blk = blk_kb * 1024
        n = (total_mb * 1024 * 1024) // blk
        src = torch.zeros(n, blk // 4, dtype=torch.float32, device=dev)
        idx = torch.randperm(n, device=dev)
        def scatter():
            _ = src.index_select(0, idx); torch.mps.synchronize()
        def contig():
            _ = src.clone(); torch.mps.synchronize()
        ts, tc = timeit(scatter), timeit(contig)
        gb = n * blk / 1e9
        out[blk_kb] = dict(block_kb=blk_kb, n_blocks=n,
                           scatter_gbps=gb/ts, contig_gbps=gb/tc, efficiency=tc/ts)
        print(f"  mps  gather blk={blk_kb:>5}KB  scatter {gb/ts:7.2f} GB/s  "
              f"contig {gb/tc:7.2f} GB/s  eff {tc/ts:.2f}", flush=True)
        del src, idx
        torch.mps.empty_cache()
    return out

# ---------------------------------------------------------------- (c) message size
def loopback(total_mb=128):
    out = {}
    for msg_kb in [4, 16, 64, 256, 1024, 4096]:
        msg = msg_kb * 1024
        n = (total_mb * 1024 * 1024) // msg
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0)); srv.listen(1)
        port = srv.getsockname()[1]
        def sink():
            c, _ = srv.accept()
            buf = bytearray(1 << 20); left = n * msg
            while left > 0:
                r = c.recv_into(buf, min(len(buf), left))
                if not r: break
                left -= r
            c.close()
        th = threading.Thread(target=sink); th.start()
        cli = socket.create_connection(("127.0.0.1", port))
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        payload = b"\0" * msg
        t = time.perf_counter()
        for _ in range(n): cli.sendall(payload)
        cli.close(); th.join()
        dt = time.perf_counter() - t
        gb = n * msg / 1e9
        out[msg_kb] = dict(msg_kb=msg_kb, n_msgs=n, gbps=gb/dt, us_per_msg=dt/n*1e6)
        print(f"  tcp  msg={msg_kb:>5}KB  {gb/dt:7.2f} GB/s  {dt/n*1e6:8.1f} us/msg", flush=True)
        srv.close()
    return out

# ------------------------------------------------- (d) RDMA model (literature constants)
def rdma_model():
    """time(msg) = overhead + bytes/BW, capped by NIC message rate.
    Constants are conservative published figures for modern RoCE/IB NICs."""
    res = {}
    for name, bw_gbps, ovh_us, msg_rate_mops in [
        ("IB NDR 400Gb", 50.0, 1.5, 20.0),
        ("RoCE 200Gb",   25.0, 2.0, 15.0),
        ("100GbE",       12.5, 3.0, 10.0),
        ("NVLink-ish",  400.0, 0.5, 50.0),
    ]:
        rows = []
        for kb in [4, 16, 64, 256, 1024, 4096]:
            b = kb * 1024
            t_wire = b / (bw_gbps * 1e9)
            t = max(t_wire + ovh_us * 1e-6, 1e-6 / msg_rate_mops)
            rows.append(dict(msg_kb=kb, eff_gbps=b/t/1e9, frac_peak=(b/t/1e9)/bw_gbps))
        res[name] = dict(peak_gbps=bw_gbps, ovh_us=ovh_us, rows=rows)
        s = "  ".join(f"{r['msg_kb']}KB:{r['frac_peak']*100:.0f}%" for r in rows)
        print(f"  {name:<14} peak {bw_gbps:5.1f} GB/s | frac of peak  {s}", flush=True)
    return res

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("(a) host scatter-gather"); a = host_gather()
    print("(b) device scatter-gather"); b = device_gather()
    print("(c) TCP loopback per-message overhead"); c = loopback()
    print("(d) RDMA analytic model"); d = rdma_model()
    json.dump(dict(host=a, device=b, tcp=c, rdma=d),
              open("results/e3_granularity.json", "w"), indent=1)
    print("wrote results/e3_granularity.json")
