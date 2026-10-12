import numpy as np

B, H, L, D, C = 1, 2, 128, 128, 64


def ref_linear_attention(q, k, v):
    h = np.zeros((B, H, D, D), dtype=np.float32)
    o = np.zeros((B, H, L, D), dtype=np.float32)
    for t in range(L):
        h += np.einsum("bhi,bhj->bhij", k[:, :, t].astype(np.float32), v[:, :, t].astype(np.float32))
        o[:, :, t] = np.einsum("bhi,bhij->bhj", q[:, :, t].astype(np.float32), h)
    return o.astype(np.float16)


def step01_numpy_sim(q, k, v):
    chunk_num = L // C
    workspace_1 = np.zeros((B, H, C, C), dtype=np.float16)
    workspace_2 = np.zeros((B, H, D, D), dtype=np.float16)
    out = np.zeros((B, H, L, D), dtype=np.float16)
    causal_mask = np.tril(np.ones((C, C), dtype=np.float32))

    # Real hardware runs `B * H` work items in parallel across cores.
    # This tutorial uses a plain sequential loop so the indexing is easy to see.
    for bz in range(B):
        for by in range(H):
            h_state = np.zeros((D, D), dtype=np.float32)
            for chunk_idx in range(chunk_num):
                l0 = chunk_idx * C
                l1 = l0 + C
                q_tile = q[bz, by, l0:l1].astype(np.float32)
                k_tile = k[bz, by, l0:l1].astype(np.float32)
                v_tile = v[bz, by, l0:l1].astype(np.float32)

                acc = (q_tile @ k_tile.T) * causal_mask
                workspace_1[bz, by] = acc.astype(np.float16)

                out[bz, by, l0:l1] = (acc @ v_tile + q_tile @ h_state).astype(np.float16)
                h_state = h_state + k_tile.T @ v_tile
                workspace_2[bz, by] = h_state.astype(np.float16)
    return out


def main():
    np.random.seed(0)
    q = np.random.randn(B, H, L, D).astype(np.float16)
    k = np.random.randn(B, H, L, D).astype(np.float16)
    v = np.random.randn(B, H, L, D).astype(np.float16)
    q = q / (np.linalg.norm(q.astype(np.float32), axis=-1, keepdims=True) + 1e-6)
    k = k / (np.linalg.norm(k.astype(np.float32), axis=-1, keepdims=True) + 1e-6)
    ref = ref_linear_attention(q, k, v)
    sim = step01_numpy_sim(q, k, v)
    np.testing.assert_allclose(sim, ref, rtol=1e-2, atol=1e-2)
    print('step01 numpy simulation passed')


if __name__ == '__main__':
    main()
