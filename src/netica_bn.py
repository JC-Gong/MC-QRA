""" Netica C-API (neticapy) 

 binning 
  -  ()  CSV  (//),  BN-Year.dne;
  -  Netica  (revise_CPTs_by_case_file)  train_fold  CPT
    (degree=1, , );  compile,  test_fold  enter_finding /
    get_beliefs  MAP  Risk Level;
  -  bn_metrics  result_netica_cv.txt

CPT  Netica ,  Python  CPT, 
 fold CSV  (=, =), 

 config/config.yaml  netica , 


    python -m src.netica_bn --config config/config.yaml --root output
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import traceback

import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["netica"]


def canonicalize_csv(src_path, dst_path):
    """ CSV  BN  (/ Netica ):
       - : ,  s ;
       - : , / ( Netica )
    , """
    def _canon(s):
        if s is None:
            return None
        s = str(s).strip()
        if s.lower() in ("", "nan", "none", "n/a", "na"):
            return None
        c = "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")
        return ("s" + c) if (c and c[0].isdigit()) else c

    with open(src_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        open(dst_path, "w", encoding="utf-8", newline="").close()
        return
    header = rows[0]
    new_header = [_canon(c) for c in header]
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        for row in rows[1:]:
            w.writerow([(_canon(v) if v != "" else "") for v in row])


def log(log_path, msg):
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_netica(cfg):
    saved = os.getcwd()
    os.chdir(cfg["bin_dir"])
    if cfg["pkg_dir"] not in sys.path:
        sys.path.insert(0, cfg["pkg_dir"])
    import neticapy
    os.chdir(saved)
    return neticapy, saved


def find_combo_dirs(root):
    return sorted(set(os.path.dirname(p) for p in
                     glob.glob(os.path.join(root, "**", "BN-Year*.dne"), recursive=True)))


def build_info(net):
    """: """
    info = {}
    for i in range(net.nodes.length):
        nd = net.nodes.get_nth_node(i)
        nm = nd.name
        states = [nd.get_state_name(j) for j in range(nd.num_states)]
        idx = {s: j for j, s in enumerate(states)}
        pars = [nd.parents.get_nth_node(p).name for p in range(nd.parents.length)]
        info[nm] = dict(node=nd, states=states, idx=idx, pars=pars, nstates=nd.num_states)
    return info


def learn_cpts(netica, net, train_csv):
    """Netica :  CPT ,  train_csv  CPT"""
    for i in range(net.nodes.length):
        net.nodes.get_nth_node(i).delete_tables()

    with open(train_csv, encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        lines = ["\t".join(cols)]
        for row in r:
            lines.append("\t".join(row.get(c, "") for c in cols))
    case_text = "\n".join(lines)

    casefile = netica.Stream("train_cas", memory=True)
    casefile.set_contents(case_text)
    net.revise_CPTs_by_case_file(casefile, net.nodes, 1)


def choose_dag(node_names, cfg):
    """ cfg['scenarios']  CSV 

     node_names , ;
     Intent/Origin  with_intent_origin
    """
    best, best_score = None, -1
    for name, dag in cfg["scenarios"].items():
        score = sum(1 for k in dag if k in node_names)
        if score > best_score:
            best, best_score = dag, score
    return best


def build_net_from_csv(netica, env, csv_path, cfg):
    """ ()  CSV  ( .dne)

    - ;  =  ();
    - :  Netica  **TAN (Tree-Augmented Naive Bayes)** ,
       CSV ( all_data) ;  cfg['structure_method'] == "fixed"
       config  scenarios  DAG
    
    """
    uniq = {}
    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        for c in cols:
            uniq[c] = set()
        for row in r:
            for c in cols:
                v = row.get(c)
                if v not in (None, ""):
                    uniq[c].add(v)
    net = netica.Net("BN_Year", None, "NO_VISUAL_INFO", environ=env)
    nd = {}
    for c in cols:
        states = sorted(uniq[c]) if uniq[c] else ["Unknown"]
        nd[c] = net.new_node(c, ",".join(states))

    
    
    method = cfg.get("structure_method", "tan")
    if method == "fixed":
        dag = choose_dag(set(cols), cfg)
        for nm, node in nd.items():
            for p in dag.get(nm, []):
                if p in nd:
                    node.add_link_from(nd[p])
    elif method == "naive":
        
        tgt_node = nd[cfg["target"]]
        for nm, node in nd.items():
            if nm != cfg["target"]:
                tgt_node.add_link_from(nd[nm])
    else:
        target = cfg["target"]
        cases = _caseset_from_csv(netica, csv_path)
        net.learn_tan_structure(net.nodes, nd[target], cases)
    return net


def _caseset_from_csv(netica, csv_path):
    """ () CSV  Netica Caseset (,  case ),
     learn_tan_structure =, =, """
    with open(csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        lines = ["\t".join(cols)]
        for row in r:
            lines.append("\t".join(row.get(c, "") for c in cols))
    stream = netica.Stream("cases_cas", memory=True)
    stream.set_contents("\n".join(lines))
    cases = netica.Caseset("AllCases")
    cases.add_cases_from_file(stream)
    return cases


def validate_combo(netica, env, combo_dir, cfg):
    target = cfg["target"]
    method = cfg.get("structure_method", "tan")
    
    result_name = f"result_netica_cv_{method}.txt"
    tmp_dir = cfg["tmp_dir"]
    log_path = os.path.join(tmp_dir, "netica_cv.log")
    n_folds = cfg.get("n_folds", 5)
    target_col = cfg.get("target_column", target)

    struct_csv = os.path.join(combo_dir, "all_data.csv")
    if not os.path.exists(struct_csv):
        struct_csv = os.path.join(combo_dir, "train_fold_1.csv")
    if not os.path.exists(struct_csv):
        log(log_path, f"[skip] {combo_dir}: missing data CSV, unable to build network structure")
        return
    net = build_net_from_csv(netica, env, struct_csv, cfg)
    info = build_info(net)
    tgt = info[target]["node"]
    n_states = info[target]["nstates"]
    state_names = info[target]["states"]

    out_lines = []
    for fold in range(1, n_folds + 1):
        train_csv = os.path.join(combo_dir, f"train_fold_{fold}.csv")
        test_csv = os.path.join(combo_dir, f"test_fold_{fold}.csv")
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
            continue
        learn_cpts(netica, net, train_csv)
        net.compile_net()

        mat = [[0] * n_states for _ in range(n_states)]
        with open(test_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                net.retract_findings()
                for nm, I in info.items():
                    if nm == target:
                        continue
                    st = I["idx"].get(row.get(nm))
                    if st is not None:
                        try:
                            I["node"].enter_finding(st)
                        except Exception:
                            pass
                beliefs = tgt.get_beliefs()
                pred = max(range(n_states), key=lambda i: beliefs[i])
                actual = info[target]["idx"].get(row.get(target_col))
                if actual is not None:
                    mat[actual][pred] += 1
        total = sum(sum(r) for r in mat)
        correct = sum(mat[i][i] for i in range(len(mat)))
        err = (1 - correct / total) * 100 if total else 0.0
        out_lines.append(f"{fold}. Confusion:\n")
        out_lines.append("  .......Predicted......\n")
        fw = max(7, max((len(s) for s in state_names), default=7) + 2)
        out_lines.append("    " + "".join(f"{s:{fw}}" for s in state_names) + "    Actual\n")
        out_lines.append("  " + "  ".join("-" * fw for _ in state_names) + "    ------\n")
        for ri, rname in enumerate(state_names):
            out_lines.append("    " + "".join(f"{mat[ri][ci]:{fw}d}" for ci in range(len(state_names)))
                             + f"    {rname}\n")
        out_lines.append(f"\nError rate = {err:.2f}%\n\n")

    with open(os.path.join(combo_dir, result_name), "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    log(log_path, f"[ok] {combo_dir} -> {result_name}")

    
    
    
    
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        learn_cpts(netica, net, struct_csv)
        net.compile_net()
        tmp_dne = os.path.join(tmp_dir, f"BN-Year_{method}.dne")
        net.write(tmp_dne)
        shutil.copy(tmp_dne, os.path.join(combo_dir, f"BN-Year_{method}.dne"))
        log(log_path, f"[dne] {combo_dir} -> BN-Year_{method}.dne")
    except Exception as e:
        log(log_path, f"[dne-fail] {combo_dir}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--root", default="output")
    args = ap.parse_args()
    cfg = load_config(args.config)

    tmp_dir = cfg["tmp_dir"]
    os.makedirs(tmp_dir, exist_ok=True)
    log_path = os.path.join(tmp_dir, "netica_cv.log")
    open(log_path, "w", encoding="utf-8").close()

    netica, saved = load_netica(cfg)
    env = netica.Environ()
    try:
        combos = find_combo_dirs(args.root)
        log(log_path, f"found {len(combos)} combos")
        for d in combos:
            try:
                validate_combo(netica, env, d, cfg)
            except Exception as e:
                log(log_path, f"[fail] {d}: {e}")
                log(log_path, traceback.format_exc())
    finally:
        env.close_netica()
        os.chdir(saved)


if __name__ == "__main__":
    main()
