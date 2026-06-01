import numpy as np


def compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    all_cmc = []
    all_ap = []

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = matches[q_idx][keep]

        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(raw_cmc)) + 1)

        ap = (precision * raw_cmc).sum() / num_rel
        all_ap.append(ap)

    all_cmc = np.asarray(all_cmc).astype(float)

    cmc = all_cmc.sum(axis=0) / len(all_cmc)
    mAP = np.mean(all_ap)

    return cmc, mAP
