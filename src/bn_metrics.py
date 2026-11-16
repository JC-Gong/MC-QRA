""" result_netica_cv.txt, 

 (group, scheme) , :
  - accuracy
  - macro_precision / macro_recall / macro_f1 ()
 mean ± std,  bn_metrics_cv_summary.csv

:
    from src.bn_metrics import summarize
    summarize("output", dataset="MC-RA", out_csv="output/bn_metrics_cv_summary.csv")
"""
import csv
import os
import re
import statistics as st
from scipy import stats as sp_stats

HEADER = [
    "group", "dataset", "method", "scheme", "bins", "labels", "n_folds",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]


def _parse_confusions(text):
    """ result_netica_cv.txt  (list[list[int]]).

    :  n (),
     n 
    """
    blocks = re.split(r"\d+\.\s*Confusion:", text)
    mats = []
    for blk in blocks[1:]:
        row_lines = []
        for line in blk.splitlines():
            nums = re.findall(r"-?\d+", line)
            if len(nums) >= 2:
                row_lines.append([int(x) for x in nums])
        if len(row_lines) < 2:
            continue
        n = len(row_lines[0])
        rows = [r for r in row_lines if len(r) == n][:n]
        if len(rows) == n:
            mats.append(rows)
    return mats


def _fold_metrics(mat):
    n = len(mat)
    total = sum(sum(r) for r in mat)
    correct = sum(mat[i][i] for i in range(n))
    acc = correct / total if total else 0.0
    precs, recs = [], []
    for c in range(n):
        tp = mat[c][c]
        col = sum(mat[r][c] for r in range(n))
        row = sum(mat[c][r] for r in range(n))
        p = tp / col if col else 0.0
        r = tp / row if row else 0.0
        precs.append(p)
        recs.append(r)
    mp = sum(precs) / n if n else 0.0
    mr = sum(recs) / n if n else 0.0
    mf = sum((2 * p * r / (p + r) if (p + r) > 0 else 0.0)
             for p, r in zip(precs, recs)) / n if n else 0.0
    return acc, mp, mr, mf


def _mean(x):
    return sum(x) / len(x) if x else 0.0


def _std(x):
    return st.stdev(x) if len(x) > 1 else 0.0


def _ci95(x):
    """95%  (),  t  (df=n-1) (lo, hi)"""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    se = _std(x) / (n ** 0.5)
    half = sp_stats.t.ppf(0.975, n - 1) * se
    m = _mean(x)
    return m - half, m + half


def _cell(x):
    """: mean±std [lo,hi] (lo,hi  95% )"""
    m, s = _mean(x), _std(x)
    lo, hi = _ci95(x)
    return f"{m:.3f}±{s:.3f} [{lo:.3f},{hi:.3f}]"


def summarize(output_dir, dataset="MC-RA",
              out_csv="output/bn_metrics_cv_summary.csv", allow=None,
              binning=None):
    rows = []
    for root, _dirs, files in os.walk(output_dir):
        
        
        result_files = [f for f in files
                        if f.startswith("result_netica_cv")
                        and f.endswith(".txt")]
        if not result_files:
            continue
        parts = root.replace(os.sep, "/").split("/")
        group = parts[-2]
        scheme = parts[-1]
        for rf in result_files:
            
            if rf == "result_netica_cv.txt":
                method = "tan"
            else:
                method = rf[len("result_netica_cv_"):-len(".txt")] or "tan"
            if allow is not None and (group, scheme) not in allow:
                continue
            bins_val, labels_val = ("", "")
            if binning and scheme in binning:
                b, l = binning[scheme]
                bins_val = "" if b is None else str(list(b))
                labels_val = "" if l is None else str(list(l))
            with open(os.path.join(root, rf), encoding="utf-8") as f:
                mats = _parse_confusions(f.read())
            if not mats:
                continue
            accs, mps, mrs, mfs = [], [], [], []
            for mat in mats:
                a, p, r, f1 = _fold_metrics(mat)
                accs.append(a)
                mps.append(p)
                mrs.append(r)
                mfs.append(f1)
            rows.append([
                group, dataset, method, scheme, bins_val, labels_val, len(mats),
                _cell(accs),
                _cell(mps),
                _cell(mrs),
                _cell(mfs),
            ])
    rows.sort(key=lambda r: (r[0], r[2], r[3]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return rows
