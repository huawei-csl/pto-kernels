import math
import numpy as np

BLOCK_DIM = 24


def ref_linear_attention(q, k, v):
    b, h, l, d = q.shape
    state = np.zeros((b, h, d, d), dtype=np.float32)
    out = np.zeros((b, h, l, d), dtype=np.float32)
    for t in range(l):
        state += np.einsum("bhi,bhj->bhij", k[:, :, t].astype(np.float32), v[:, :, t].astype(np.float32))
        out[:, :, t] = np.einsum("bhi,bhij->bhj", q[:, :, t].astype(np.float32), state)
    return out.astype(np.float16)


def step02_numpy_sim(q, k, v, chunk_size):
    b, h, l, d = q.shape
    total_work = b * h
    chunk_num = l // chunk_size
    workspace_1 = np.zeros((BLOCK_DIM, chunk_size, chunk_size), dtype=np.float16)
    workspace_2 = np.zeros((BLOCK_DIM, d, d), dtype=np.float16)
    out = np.zeros((b, h, l, d), dtype=np.float16)
    causal_mask = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32))

    # The real kernel launches one fixed block per core and loops over `work_idx` inside the kernel.
    # We emulate that with a sequential nested loop instead of actual parallel execution.
    for work_idx in range(math.ceil(total_work / BLOCK_DIM)):
        for cid in range(BLOCK_DIM):
            pid = work_idx * BLOCK_DIM + cid
            if pid >= total_work:
                continue
            by = pid % h
            bz = pid // h
            h_state = np.zeros((d, d), dtype=np.float32)
            for chunk_idx in range(chunk_num):
                l0 = chunk_idx * chunk_size
                l1 = l0 + chunk_size
                q_tile = q[bz, by, l0:l1].astype(np.float32)
                k_tile = k[bz, by, l0:l1].astype(np.float32)
                v_tile = v[bz, by, l0:l1].astype(np.float32)

                acc = (q_tile @ k_tile.T) * causal_mask
                workspace_1[cid] = acc.astype(np.float16)

                out[bz, by, l0:l1] = (acc @ v_tile + q_tile @ h_state).astype(np.float16)
                h_state = h_state + k_tile.T @ v_tile
                workspace_2[cid] = h_state.astype(np.float16)
    return out


def main():
    np.random.seed(0)
    for shape in [(1, 2, 256, 128, 64), (4, 2, 512, 128, 64)]:
        b, h, l, d, c = shape
        q = np.random.randn(b, h, l, d).astype(np.float16)
        k = np.random.randn(b, h, l, d).astype(np.float16)
        v = np.random.randn(b, h, l, d).astype(np.float16)
        q = q / (np.linalg.norm(q.astype(np.float32), axis=-1, keepdims=True) + 1e-6)
        k = k / (np.linalg.norm(k.astype(np.float32), axis=-1, keepdims=True) + 1e-6)
        ref = ref_linear_attention(q, k, v)
        sim = step02_numpy_sim(q, k, v, c)
        np.testing.assert_allclose(sim, ref, rtol=1e-2, atol=1e-2)
        print('passed', shape)


if __name__ == '__main__':
    main()
