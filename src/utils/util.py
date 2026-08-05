import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support
)

def find_transitions(seq: np.ndarray):
    seq = np.asarray(seq)
    idx = np.where(seq[1:] != seq[:-1])[0] + 1
    return idx

def run_lengths(seq: np.ndarray):
    seq = np.asarray(seq)
    if seq.size == 0:
        return {}
    rl = defaultdict(list)
    cur = seq[0]
    cnt = 1
    for x in seq[1:]:
        if x == cur:
            cnt += 1
        else:
            rl[cur].append(cnt)
            cur = x
            cnt = 1
    rl[cur].append(cnt)
    return rl

# ---------- 1) False Transition Rate (FTR) ----------

def false_transition_rate(y_true, y_pred, tolerance: int = 0):
    """
    FTR = (# 预测切换点中，无法与任意真切换点在 ±tolerance 范围内匹配的数量) / (# 预测切换点)
    若预测没有切换点，返回 0.0（无“误报”可言）。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_tr = find_transitions(y_pred)
    true_tr = find_transitions(y_true)

    if pred_tr.size == 0:
        return 0.0

    matched = np.zeros(pred_tr.size, dtype=bool)
    if true_tr.size > 0:
        true_tr_sorted = np.sort(true_tr)
        for i, p in enumerate(pred_tr):
            # 在真切换点里找是否有 |t - p| <= tolerance 的
            # 用二分近邻以提升效率
            j = np.searchsorted(true_tr_sorted, p)
            candidates = []
            if j < true_tr_sorted.size:
                candidates.append(true_tr_sorted[j])
            if j > 0:
                candidates.append(true_tr_sorted[j-1])
            for t in candidates:
                if abs(int(t) - int(p)) <= tolerance:
                    matched[i] = True
                    break

    false_preds = np.sum(~matched)
    return false_preds / pred_tr.size

# ---------- 2) Edit Distance (Levenshtein) ----------

def edit_distance(y_true, y_pred, normalize: bool = False):
    """
    纯 Python/NumPy 的编辑距离 DP 实现。
    若 normalize=True，返回 距离 / max(len(y_true), len(y_pred))
    """
    a = np.asarray(y_true)
    b = np.asarray(y_pred)
    n, m = len(a), len(b)
    # DP 矩阵 (n+1) x (m+1)
    dp = np.zeros((n+1, m+1), dtype=int)
    dp[0, :] = np.arange(m+1)
    dp[:, 0] = np.arange(n+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,      # 删除
                dp[i, j-1] + 1,      # 插入
                dp[i-1, j-1] + cost  # 替换
            )
    dist = int(dp[n, m])
    if normalize:
        denom = max(n, m) if max(n, m) > 0 else 1
        return dist / denom
    return dist

# ---------- 3) Run-length 与 RL-EMD ----------

def _hist_from_run_lengths(rl_list, max_len: int):
    """
    将一组连续段长度映射为长度直方图（1..max_len，超过的都计入 max_len 桶）。
    返回归一化直方图（和为1）；若无数据返回全零向量。
    """
    hist = np.zeros(max_len, dtype=float)
    for L in rl_list:
        bin_idx = min(L, max_len) - 1  # 长度1落到索引0
        hist[bin_idx] += 1
    s = hist.sum()
    return hist / s if s > 0 else hist

def _emd_1d(p, q):
    """
    1D Earth Mover's Distance 等于累计分布差之 L1 范数。
    这里 p, q 均为概率直方图（和为1）。
    """
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return np.sum(np.abs(cdf_p - cdf_q))

def run_length_stats(y_true, y_pred):
    """
    返回两个字典：
        rl_true[label] = [run lengths...]
        rl_pred[label] = [run lengths...]
    方便你自己做可视化或统计（均值/中位数等）。
    """
    return run_lengths(np.asarray(y_true)), run_lengths(np.asarray(y_pred))

def rl_emd(y_true, y_pred, classes=None, max_len: int = 20, weight_by_support: bool = True):
    """
    计算基于“连续段长度分布”的 EMD，逐类求 EMD 后做加权平均。
    - classes: 指定所有可能类的有序列表；None 则从 y_true ∪ y_pred 中推断
    - max_len: 连续段长度分布的最大桶（≥max_len 的并入最后一桶）
    - weight_by_support: 是否按真值中该类的 run 段数占比加权（推荐 True）
    返回: overall_emd, per_class_emd(dict)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    rl_t = run_lengths(y_true)
    rl_p = run_lengths(y_pred)

    # 统计各类在真值中的 run 段数（用于加权）
    support = {c: len(rl_t.get(c, [])) for c in classes}
    total_support = max(sum(support.values()), 1)

    per_class_emd = {}
    weights = {}
    for c in classes:
        ht = _hist_from_run_lengths(rl_t.get(c, []), max_len)
        hp = _hist_from_run_lengths(rl_p.get(c, []), max_len)
        emd = _emd_1d(ht, hp)
        per_class_emd[c] = emd
        weights[c] = (support[c] / total_support) if weight_by_support else (1.0 / len(classes))

    overall = float(sum(per_class_emd[c] * weights[c] for c in classes))
    return overall, per_class_emd

# ---------- 便捷封装：一次性算三项 ----------

def sequence_structure_metrics(
    y_true, y_pred,
    ftr_tolerance: int = 0,
    edit_norm: bool = True,
    rl_max_len: int = 20,
    rl_weight_by_support: bool = True,
):
    """
    返回一个 dict，包含：
      - ftr
      - edit_distance / edit_distance_norm
      - rl_emd_overall, rl_emd_per_class, rl_true, rl_pred（便于画图/表）
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true 与 y_pred 长度必须一致"

    ftr = false_transition_rate(y_true, y_pred, tolerance=ftr_tolerance)
    ed = edit_distance(y_true, y_pred, normalize=edit_norm)
    rl_true, rl_pred = run_length_stats(y_true, y_pred)
    rl_overall, rl_pc = rl_emd(y_true, y_pred, max_len=rl_max_len, weight_by_support=rl_weight_by_support)

    # 做归一化，避免 max_len 影响数值范围
    rl_norm = rl_overall / (rl_max_len - 1)
    rl_pc_norm = {c: v / (rl_max_len - 1) for c, v in rl_pc.items()}

    return {
        "ftr": ftr,
        "edit_distance_norm" if edit_norm else "edit_distance": ed,
        "rl_emd_overall": rl_overall,  # 原始值
        "rl_emd_overall_norm": rl_norm,  # 归一化到 [0,1]
        "rl_emd_per_class": rl_pc,  # 原始分阶段
        "rl_emd_per_class_norm": rl_pc_norm,  # 分阶段归一化
        "rl_true": rl_true,
        "rl_pred": rl_pred,
    }

def _fmt(x, nd = 4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _as_np1d(x):
    arr = np.asarray(x)
    return arr.astype(float).reshape(-1)

def _safe_avg(x, w=None):
    x = _as_np1d(x)
    if w is None:
        return float(np.mean(x)) if x.size else float("nan")
    w = _as_np1d(w)
    if x.size == 0 or w.size == 0 or np.sum(w) == 0:
        return float("nan")
    return float(np.average(x, weights=w))

def make_masks_one(overlap, L):
    dl = max(0, min((overlap-1), L))
    dr = max(0, min((overlap-1), L - dl))
    center_mask_one = torch.zeros(L, dtype=torch.bool)
    center_mask_one[dl: L - dr] = True
    pair_mask_one = center_mask_one[:-1] & center_mask_one[1:]
    return center_mask_one, pair_mask_one

def expand_mask(mask_one, B):
    return mask_one.unsqueeze(0).expand(B, -1).to(torch.bool)

def to_cuda(gs, device = torch.device("cuda")):
    if torch.cuda.is_available():
        if type(gs) == list:
            return [g.to(device) for g in gs]
        return gs.to(device)
    return gs

def compute_metrics(y_true, y_pred, class_names=('Wake','N1','N2','N3','Rem')):
    f1 = f1_score(y_true, y_pred, average='weighted')
    bacc = balanced_accuracy_score(y_true, y_pred)
    C = len(class_names)
    labels = list(range(C))
    _, _, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return bacc, f1_cls, f1

def masked_mean(x, mask):
    if mask.any():
        return x[mask].mean()
    else:
        return torch.zeros((), device=x.device)

def set_device(config):
    if torch.cuda.is_available() and config.cuda is not None:
        torch.cuda.set_device(config.cuda)
        device = torch.device("cuda")
        print(
            f"Using CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device

def dl2_logic_loss(probs, pair_mask_B):
    W, N1, N2, N3, REM = 0, 1, 2, 3, 4
    p = probs
    B, L, S = p.shape
    device = p.device
    p_t = p[:, :-1, :]
    p_tp1 = p[:, 1:, :]
    pmask = pair_mask_B.bool()

    P = p_t.unsqueeze(-1) * p_tp1.unsqueeze(-2)
    A_pos = torch.zeros(5, 5, device=device)
    A_pos[W, N3] = 1
    A_pos[W, REM] = 0.5
    A_pos[N1, N3] = 1
    A_pos[N3, N1] = 0.5
    A_pos[N3, REM] = 1
    A_pos[REM, N2] = 0.5
    A_pos[REM, N3] = 1

    A_neg = torch.zeros(5, 5, device=device)
    A_neg[W, N1] = 0.2
    A_neg[N1, W] = 0.2
    A_neg[N1, N2] = 0.6
    A_neg[N2, N1] = 0.2
    A_neg[N2, N3] = 0.1
    A_neg[N2, W] = 0.1
    A_neg[N3, N2] = 0.1
    A_neg[REM, W] = 0.1
    A_neg[REM, N1] = 0.1

    penalty = masked_mean((P * A_pos).view(B, L - 1, -1), pmask)
    reward = masked_mean((P * A_neg).view(B, L - 1, -1), pmask)

    stage_K = {2: 3, 3: 5, 4: 3}
    def alpha_k(k):
        return 0.6 ** (k - 1)

    stage_w = {2: 1.6, 3: 1.9, 4: 1.2}
    dwell_reward = torch.zeros((), device=device)
    for s in (2, 3, 4):
        K = stage_K[s]
        w_s = stage_w[s]
        p_s = p[..., s]
        stage_reward = torch.zeros((), device=device)
        valid_terms = 0
        for k in range(1, K + 1):
            if p_s.shape[1] - k <= 0:
                break
            stay_k = p_s[:, :-k] * p_s[:, k:]
            Lm1 = pmask.shape[1]
            Lk = Lm1 - (k - 1)
            if Lk <= 0:
                continue
            m_k = pmask[:, :Lk].clone()
            for r in range(1, k):
                m_k = m_k & pmask[:, r:r + Lk]
            if m_k.any():
                stage_reward = stage_reward + alpha_k(k) * masked_mean(stay_k, m_k)
                valid_terms += 1
        if valid_terms > 0:
            dwell_reward = dwell_reward + w_s * (stage_reward / valid_terms)
    dwell_loss = -dwell_reward

    L_logic = penalty + dwell_loss - reward
    return L_logic

def entropy_on_centers(logits_seq, center_mask_B):
    probs = logits_seq.softmax(-1)
    ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(-1)
    return ent[center_mask_B.bool()].mean()