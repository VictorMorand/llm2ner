from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


@dataclass
class Span:
    start: int
    end: int
    score: float


@dataclass
class DiscontinuousSpan:
    start: int
    a: int
    b: int
    end: int
    score: float


def extract_scores(span_probs: np.ndarray) -> np.ndarray:
    assert span_probs.ndim == 3, "span_probs must be [1, n, n]"
    S = span_probs[0].copy()
    return S.T


# ------------ Overlap counting (violated non-overlap constraint) ------------
def count_overlaps(spans: List[Tuple[int, int]]) -> int:
    """
    Count the number of overlapping span pairs in a list of (start,end) with inclusive indices.
    Overlap means the intersection is non-empty.

    Runs in O(m log m) using a sweep.
    """
    events = []
    for i, j in spans:
        # start event at i, end event just after j (j+1) so inclusive endpoints don't cancel overlap
        events.append((i, +1))
        events.append((j + 1, -1))
    events.sort()

    active = 0
    violations = 0
    for _, delta in events:
        if delta == +1:
            # This new span overlaps with all currently active ones
            violations += active
            active += 1
        else:
            active -= 1
    return violations


# ---------------- Decoding 1: flat non-overlapping NER ----------------
def weighted_interval_scheduling(spans: List[Span]) -> List[Span]:
    spans_sorted = sorted(spans, key=lambda s: (s.end, s.start))
    ends = [s.end for s in spans_sorted]
    p = []
    for j, sj in enumerate(spans_sorted):
        pred = -1
        lo, hi = 0, j - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if ends[mid] < sj.start:
                pred = mid
                lo = mid + 1
            else:
                hi = mid - 1
        p.append(pred)
    n = len(spans_sorted)
    M = [0.0] * (n + 1)
    choose = [False] * n
    for j in range(1, n + 1):
        opt1 = M[j - 1]
        wj = spans_sorted[j - 1].score
        pj = p[j - 1]
        opt2 = wj + (M[pj + 1] if pj >= 0 else 0.0)
        if opt2 > opt1:
            M[j] = opt2
            choose[j - 1] = True
        else:
            M[j] = opt1
            choose[j - 1] = False
    res = []
    j = n
    while j > 0:
        if choose[j - 1]:
            sj = spans_sorted[j - 1]
            res.append(sj)
            pj = p[j - 1]
            j = pj + 1
        else:
            j -= 1
    res.reverse()
    return res


def decode_flat(S: np.ndarray, min_score: float = 0.0) -> List[Span]:
    n = S.shape[0]
    spans = []
    for i in range(n):
        for j in range(i, n):
            sc = float(S[i, j])
            if sc > min_score:
                spans.append(Span(i, j, sc))
    return weighted_interval_scheduling(spans)


# ---------------- Decoding 2: nested NER ----------------
def decode_well_nested(S: np.ndarray, min_score: float = 0.0) -> List[Span]:
    n = S.shape[0]
    DP = [[0.0] * n for _ in range(n)]
    BP: List[List[Optional[Tuple[str, int]]]] = [[None] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            best = DP[i + 1][j] if i + 1 <= j else 0.0
            bp = ("skip", -1)
            for k in range(i, j + 1):
                sc = float(S[i, k])
                if sc < min_score:
                    continue
                left = DP[i + 1][k - 1] if i + 1 <= k - 1 else 0.0
                right = DP[k + 1][j] if k + 1 <= j else 0.0
                tot = sc + left + right
                if tot > best:
                    best = tot
                    bp = ("take", k)
            DP[i][j] = best
            BP[i][j] = bp
    res: List[Span] = []

    def rec(i: int, j: int):
        if i > j:
            return
        tag, k = BP[i][j]
        if tag == "skip":
            if i + 1 <= j:
                rec(i + 1, j)
        elif tag == "take":
            res.append(Span(i, k, float(S[i, k])))
            if i + 1 <= k - 1:
                rec(i + 1, k - 1)
            if k + 1 <= j:
                rec(k + 1, j)

    rec(0, n - 1)
    res.sort(key=lambda s: (s.start, s.end))
    return res


# ---------------- Decoding 3: discontinuous (max 1 gap) nested NER ----------------
def decode_discontinuous_nested(
    S: np.ndarray, min_score: float = 0.0, gap_penalty: float = 0.2
):
    n = S.shape[0]
    DP = [[0.0] * n for _ in range(n)]
    BP: List[List[Optional[Tuple[str, Tuple]]]] = [[None] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            best = DP[i + 1][j] if i + 1 <= j else 0.0
            bp: Optional[Tuple[str, Tuple]] = ("skip", ())
            for k in range(i, j + 1):
                sc = float(S[i, k])
                if sc >= min_score:
                    left = DP[i + 1][k - 1] if i + 1 <= k - 1 else 0.0
                    right = DP[k + 1][j] if k + 1 <= j else 0.0
                    tot = sc + left + right
                    if tot > best:
                        best = tot
                        bp = ("cont", (i, k))
            for k in range(i + 1, j + 1):
                for a in range(i, k):
                    for b in range(a + 1, k + 1):
                        sc = float(S[i, a]) + float(S[b, k]) - gap_penalty
                        if sc < min_score:
                            continue
                        p1 = DP[i + 1][a - 1] if i + 1 <= a - 1 else 0.0
                        mid = DP[a + 1][b - 1] if a + 1 <= b - 1 else 0.0
                        p2 = DP[b + 1][k - 1] if b + 1 <= k - 1 else 0.0
                        right = DP[k + 1][j] if k + 1 <= j else 0.0
                        tot = sc + p1 + mid + p2 + right
                        if tot > best:
                            best = tot
                            bp = ("disc", (i, a, b, k))
            DP[i][j] = best
            BP[i][j] = bp
    cont: List[Span] = []
    disc: List[DiscontinuousSpan] = []

    def rec(i: int, j: int):
        if i > j:
            return
        tag_tuple = BP[i][j]
        if tag_tuple is None:
            return
        tag, args = tag_tuple
        if tag == "skip":
            if i + 1 <= j:
                rec(i + 1, j)
        elif tag == "cont":
            ii, kk = args
            cont.append(Span(ii, kk, float(S[ii, kk])))
            if ii + 1 <= kk - 1:
                rec(ii + 1, kk - 1)
            if kk + 1 <= j:
                rec(kk + 1, j)
        elif tag == "disc":
            ii, aa, bb, kk = args
            disc.append(
                DiscontinuousSpan(
                    ii, aa, bb, kk, float(S[ii, aa] + S[bb, kk] - gap_penalty)
                )
            )
            if ii + 1 <= aa - 1:
                rec(ii + 1, aa - 1)
            if aa + 1 <= bb - 1:
                rec(aa + 1, bb - 1)
            if bb + 1 <= kk - 1:
                rec(bb + 1, kk - 1)
            if kk + 1 <= j:
                rec(kk + 1, j)

    rec(0, n - 1)
    cont.sort(key=lambda s: (s.start, s.end))
    disc.sort(key=lambda s: (s.start, s.a, s.b, s.end))
    return cont, disc


# ---------------- Decoding 4: threshold-based overlapping NER ----------------
def decode_threshold(S: np.ndarray, min_score: float = 0.5) -> List[Span]:
    """Return ALL spans with score > min_score, without enforcing any constraints.
    This can (and will) return overlapping and crossing spans.
    """
    n = S.shape[0]
    out: List[Span] = []
    for i in range(n):
        for j in range(i, n):
            sc = float(S[i, j])
            if sc > min_score:
                out.append(Span(i, j, sc))
    out.sort(key=lambda s: (-s.score, s.start, s.end))
    return out


def spans_from_bio(tags: List[int]) -> List[Tuple[int, int]]:
    spans = []
    i = 0
    n = len(tags)
    while i < n:
        if tags[i] == 1:
            j = i
            while j + 1 < n and tags[j + 1] == 2:
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


# ---------------- Decoding 5: Greedy (non-overlapping) ----------------
def _overlaps(i: int, j: int, picked: List["Span"]) -> bool:
    # overlap if intervals intersect (inclusive indices)
    for s in picked:
        if not (j < s.start or i > s.end):
            return True
    return False


def decode_greedy(
    S: np.ndarray, min_score: float = 0.0, prefer_long: bool = True
) -> List[Span]:
    """
    Greedy selection of non-overlapping spans:
      1) list all spans with score > min_score
      2) sort by score desc; tie-break by length (desc if prefer_long else asc), then by (start,end)
      3) pick spans that don't overlap with those already chosen
    Args: 
        S: [n,n] array of span scores
        min_score: minimum score to consider a span
        prefer_long: if True, prefer longer spans in tie-breaks; else prefer shorter spans
    """
    n = S.shape[0]
    cands: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i, n):
            sc = float(S[i, j])
            if sc > min_score:
                cands.append((sc, i, j))

    # tie-breakers: score desc, length desc (or asc), then start, then end
    if prefer_long:
        cands.sort(key=lambda t: (-t[0], -(t[2] - t[1] + 1), t[1], t[2]))
    else:
        cands.sort(key=lambda t: (-t[0], (t[2] - t[1] + 1), t[1], t[2]))

    picked: List[Span] = []
    for sc, i, j in cands:
        if not _overlaps(i, j, picked):
            picked.append(Span(i, j, sc))

    picked.sort(key=lambda s: (s.start, s.end))
    return picked


def line_col_greedy(span_probs, threshold=0.5):
    
    # (1) Row-wise argmax
    max_lines = np.argmax(span_probs, axis=1)  # indices of max per row
    mask_rows = np.zeros_like(span_probs, dtype=bool)
    mask_rows[np.arange(span_probs.shape[0]), max_lines] = True
    span_probs[~mask_rows] = 0

    # (2) Column-wise argmax
    max_cols = np.argmax(span_probs, axis=0)  # indices of max per column
    mask_cols = np.zeros_like(span_probs, dtype=bool)
    mask_cols[max_cols, np.arange(span_probs.shape[1])] = True
    span_probs[~mask_cols] = 0

    # Extract spans
    msk = span_probs > threshold
    spans = np.argwhere(msk)
    scores = span_probs[msk]

    return [Span(int(start), int(end), float(score)) for (start, end), score in zip(spans, scores)]


def gold_spans(item: Dict[str, Any]) -> List[Tuple[int, int]]:
    if "ent_pos" in item and item["ent_pos"]:
        return [(int(a), int(b)) for (a, b) in item["ent_pos"]]
    for key in ("token_tags", "ner_tags"):
        if key in item and item[key]:
            return spans_from_bio(list(item[key]))
    return []



# ---------------- Decoding 6: Greedy Nested (non-crossing; nesting allowed) ----------------

def _noncrossing_with_all(i: int, j: int, picked: List["Span"]) -> bool:
    """
    Returns True iff [i,j] is non-crossing w.r.t. all spans in `picked`.
    Allowed relations with every picked span s = [a,b]:
      - disjoint: j < a or i > b
      - contained: a <= i and j <= b
      - contains:  i <= a and b <= j
    Anything else is a crossing and is rejected.
    """
    for s in picked:
        a, b = s.start, s.end
        if j < a or i > b:
            continue  # disjoint
        if (a <= i and j <= b) or (i <= a and b <= j):
            continue  # nested (contained or contains)
        return False  # crossing
    return True

def decode_greedy_nested(S: np.ndarray, min_score: float = 0.0, prefer_long: bool = True) -> List[Span]:
    """
    Greedy, non-crossing (well-nested) decoding:
      1) collect all spans with score > min_score
      2) sort by score desc; tie-break by length (desc if prefer_long else asc), then (start,end)
      3) add a span iff it is non-crossing with all already picked spans (nesting allowed)
    """
    n = S.shape[0]
    cands: List[Tuple[float,int,int]] = []
    for i in range(n):
        for j in range(i, n):
            sc = float(S[i, j])
            if sc > min_score:
                cands.append((sc, i, j))

    if prefer_long:
        cands.sort(key=lambda t: (-t[0], -(t[2]-t[1]+1), t[1], t[2]))
    else:
        cands.sort(key=lambda t: (-t[0],  (t[2]-t[1]+1), t[1], t[2]))

    picked: List[Span] = []
    for sc, i, j in cands:
        if _noncrossing_with_all(i, j, picked):
            picked.append(Span(i, j, sc))

    picked.sort(key=lambda s: (s.start, s.end))
    return picked



# ---------------- Evaluation ----------------
METHODS = [
    "flat", 
    "well_nested", 
    "disc_cont", 
    "threshold", 
    "greedy", 
    "greedy_nested", 
    "line_col_greedy"
    ]

def decode_all(
    item: Dict[str, Any], min_score=0.05, gap_penalty=0.2, threshold_for_all=0.5
):
    S = extract_scores(np.asarray(item["span_probs"], dtype=np.float32))
    cont, disc = decode_discontinuous_nested(
        S, min_score=min_score, gap_penalty=gap_penalty
    )
    return {
        "flat": decode_flat(S, min_score=min_score),
        "well_nested": decode_well_nested(S, min_score=min_score),
        "disc_cont": cont + [Span(s.start, s.end, s.score) for s in disc],
        "threshold": decode_threshold(S, min_score=threshold_for_all),
        "greedy": decode_greedy(S, min_score=min_score, prefer_long=True),
        "line_col_greedy": line_col_greedy(S, threshold=threshold_for_all),
        "greedy_nested": decode_greedy_nested(S, min_score=min_score, prefer_long=True),
    }, disc


def decode(
    span_probs: torch.Tensor,
    decoding_strategy: str = "greedy",
    threshold=0.5,
    gap_penalty=0.2,
) -> List[Span]:
    """Decode span scores S using the specified decoding strategy.
    Args:
        span_probs: [1,n,n] tensor of span scores
        decoding_strategy: one of "flat", "well_nested", "disc_cont", "threshold", "greedy"
        threshold: minimum score threshold (used in some strategies)
        gap_penalty: gap penalty for discontinuous decoding (only used if decoding_strategy=="disc_cont")
    Returns:
        List of decoded Span objects.
    """
    S = extract_scores(span_probs.detach().cpu().numpy())
    if decoding_strategy == "flat":
        spans = decode_flat(S, min_score=threshold)
    elif decoding_strategy == "well_nested":
        spans = decode_well_nested(S, min_score=threshold)
    elif decoding_strategy == "disc_cont":
        cont, disc = decode_discontinuous_nested(
            S, min_score=threshold, gap_penalty=gap_penalty
        )
        spans = cont + [Span(s.start, s.end, s.score) for s in disc]
    elif decoding_strategy == "threshold":
        spans = decode_threshold(S, min_score=threshold)
    elif decoding_strategy == "greedy":
        spans = decode_greedy(S, min_score=threshold, prefer_long=True)
    elif decoding_strategy == "line_col_greedy" or decoding_strategy == "greedy_nested":
        spans = line_col_greedy(S, threshold=threshold)
    else:
        raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
    
    return spans


def to_set(spans: List[Tuple[int, int]]):
    return set(spans)


def eval_dataset(
    dataset: List[Dict[str, Any]], min_score=0.5, gap_penalty=0.2, score_threshold=0.5, limit_samples: int = None
) -> pd.DataFrame:
    if limit_samples is not None:
        dataset = dataset[:limit_samples]
    stats = {a: {"TP": 0, "FP": 0, "FN": 0} for a in METHODS}
    overlaps = {a: 0 for a in METHODS}
    disc_pick_count = 0
    for item in tqdm(dataset, desc="Evaluating"):
        gold = gold_spans(item)
        G = to_set(gold)
        decoded_all, disc = decode_all(
            item, min_score, gap_penalty, score_threshold
        )
        L = {method: [(s.start, s.end) for s in decoded_all[method]] for method in METHODS}
        
        # Count overlaps per algorithm
        for a in L:
            overlaps[a] += count_overlaps(L[a])
        # Sets for TP/FP/FN computation
        P = {a: set(L[a]) for a in L}
        disc_pick_count += len(disc)
        for a in METHODS:
            TP = len(G & P[a])
            FP = len(P[a] - G)
            FN = len(G - P[a])
            stats[a]["TP"] += TP
            stats[a]["FP"] += FP
            stats[a]["FN"] += FN
    rows = []
    for a in METHODS:
        TP, FP, FN = stats[a]["TP"], stats[a]["FP"], stats[a]["FN"]
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append(
            {
                "algorithm": a,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "overlap_violations": overlaps[a],
                "disc_picks": disc_pick_count if a == "disc_cont" else 0,
                "greedy_picks": len(P["greedy"]) if a == "greedy" else 0,
            }
        )
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return df
