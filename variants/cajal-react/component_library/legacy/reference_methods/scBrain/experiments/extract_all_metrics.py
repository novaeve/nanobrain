#!/usr/bin/env python3
"""Extract test classification metrics from ALL experiment log files."""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("./experiments")

# ── helpers ──────────────────────────────────────────────────────────────────

def find_all_log_files():
    """Find all .log and nohup.out files recursively."""
    logs = []
    for root, dirs, files in os.walk(BASE):
        for f in files:
            if f.endswith(".log") or f == "nohup.out":
                logs.append(os.path.join(root, f))
    return sorted(logs)


def parse_metrics_from_log(filepath):
    """Parse test classification metrics from a log file.

    Returns dict of {level: {metric_name: value}} for the LAST test run found.
    Levels: CLASS, SUBCLASS, SUPERTYPE (and variants like RETRAINED_*, PROJECTION_*, OT_*)
    """
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    results = {}
    # We'll collect ALL occurrences and keep the last one per prefix+level

    # Patterns for macro/micro lines:
    # [CLASS] Macro  - Precision: 0.9731 | Recall: 0.9691 | F1: 0.9710
    # [CLASS] Macro  - P: 0.9731 | R: 0.9691 | F1: 0.9710
    # [RETRAINED_CLASS] Macro  - P: 0.9478 | R: 0.9301 | F1: 0.9385
    # [PROJECTION_CLASS] Macro  - P: ...
    # [OT_CLASS] Macro  - P: ...

    macro_pat = re.compile(
        r'\[(\w+)\]\s+Macro\s+-\s+(?:Precision|P):\s*([\d.]+)\s*(?:\|\s*(?:Recall|R):\s*([\d.]+))?\s*(?:\|\s*F1:\s*([\d.]+))?'
    )
    # Also handle no-space variant: P:0.8628 R:0.8680 F1:0.8648
    macro_pat2 = re.compile(
        r'\[(\w+)\]\s+Macro\s+-\s+P:([\d.]+)\s+R:([\d.]+)\s+F1:([\d.]+)'
    )
    micro_pat = re.compile(
        r'\[(\w+)\]\s+Micro\s+-\s+(?:Precision|P):\s*([\d.]+)\s*(?:\|\s*(?:Recall|R):\s*([\d.]+))?\s*(?:\|\s*F1:\s*([\d.]+))?'
    )
    micro_pat2 = re.compile(
        r'\[(\w+)\]\s+Micro\s+-\s+P:([\d.]+)\s+R:([\d.]+)\s+F1:([\d.]+)'
    )
    # Micro with only F1: [LEVEL0] Micro  - F1:0.9122
    micro_pat3 = re.compile(
        r'\[(\w+)\]\s+Micro\s+-\s+F1:([\d.]+)'
    )
    auc_pat = re.compile(
        r'\[(\w+)\]\s+AUC-ROC\s+-\s+Macro:\s*([\d.]+)\s*\|\s*Micro:\s*([\d.]+)'
    )
    # No-space variant: Macro:0.9924 Micro:0.9936
    auc_pat2 = re.compile(
        r'\[(\w+)\]\s+AUC-ROC\s+-\s+Macro:([\d.]+)\s+Micro:([\d.]+)'
    )

    for line in lines:
        # Try no-space patterns first (kukanja format: P:0.7042 R:0.6908 F1:0.6866)
        m = macro_pat2.search(line)
        if not m:
            m = macro_pat.search(line)
        if m:
            key = m.group(1)
            results.setdefault(key, {})
            results[key]["macro_p"] = float(m.group(2))
            if m.group(3):
                results[key]["macro_r"] = float(m.group(3))
            if m.group(4):
                results[key]["macro_f1"] = float(m.group(4))
            continue
        m = micro_pat2.search(line)
        if not m:
            m = micro_pat.search(line)
        if m:
            key = m.group(1)
            results.setdefault(key, {})
            results[key]["micro_p"] = float(m.group(2))
            if m.group(3):
                results[key]["micro_r"] = float(m.group(3))
            if m.group(4):
                results[key]["micro_f1"] = float(m.group(4))
            continue
        m = micro_pat3.search(line)
        if m:
            key = m.group(1)
            results.setdefault(key, {})
            results[key]["micro_f1"] = float(m.group(2))
            continue
        m = auc_pat2.search(line)
        if not m:
            m = auc_pat.search(line)
        if m:
            key = m.group(1)
            results.setdefault(key, {})
            results[key]["auc_macro"] = float(m.group(2))
            results[key]["auc_micro"] = float(m.group(3))
            continue

    return results


def normalize_level(key):
    """Map RETRAINED_CLASS -> CLASS, PROJECTION_CLASS -> CLASS, OT_CLASS -> CLASS, etc.
    Also map kukanja levels: ANNOTATION_BROAD -> CLASS, ANNOTATION_MEDIUM -> SUBCLASS,
    ANNOTATION_HIGH -> SUPERTYPE, ANNOTATION_MEDIUM_DA -> SUBCLASS_DA.
    Also map LEVEL0 -> CLASS, LEVEL1_5 -> SUBCLASS, LEVEL2 -> SUBCLASS_DA, LEVEL3 -> SUPERTYPE."""
    # Kukanja level mapping
    kukanja_map = {
        "ANNOTATION_BROAD": ("", "CLASS"),
        "ANNOTATION_MEDIUM": ("", "SUBCLASS"),
        "ANNOTATION_MEDIUM_DA": ("", "SUBCLASS_DA"),
        "ANNOTATION_HIGH": ("", "SUPERTYPE"),
        "LEVEL0": ("", "CLASS"),
        "LEVEL1_5": ("", "SUBCLASS"),
        "LEVEL2": ("", "SUBCLASS_DA"),
        "LEVEL3": ("", "SUPERTYPE"),
    }
    if key in kukanja_map:
        return kukanja_map[key]

    for prefix in ("RETRAINED_", "PROJECTION_", "OT_"):
        if key.startswith(prefix):
            return prefix.rstrip("_"), key[len(prefix):]
    return "", key


def check_diverged(results):
    """Check if SUPERTYPE F1 < 0.001."""
    for key in results:
        _, level = normalize_level(key)
        if level == "SUPERTYPE":
            f1 = results[key].get("macro_f1", 0)
            if f1 < 0.001:
                return True
    return False


def check_still_training(filepath):
    """Check if a log file is still training (no test metrics, has epoch lines)."""
    with open(filepath, "r", errors="replace") as f:
        content = f.read()
    has_epoch = bool(re.search(r"Epoch \[\d+/\d+\]", content))
    has_test = bool(re.search(r"est [Cc]lassification|Macro\s+-\s+(?:P|Precision):", content))
    return has_epoch and not has_test


def derive_display_name(filepath):
    """Derive a human-readable name from the log path."""
    rel = os.path.relpath(filepath, BASE)
    parts = Path(rel).parts

    # Handle kukanja subdirs
    if parts[0] == "kukanja" and len(parts) >= 3:
        dataset = parts[1]  # Kukanja_EAE or Kukanja_MS
        fname = parts[2]
        fname = re.sub(r"^record_", "", fname)
        fname = re.sub(r"_s\d+\.log$", "", fname)
        return f"kukanja/{dataset}/{fname}"

    # Handle eval_lr
    if parts[0] == "eval_lr":
        fname = parts[-1]
        fname = re.sub(r"^record_lr_", "", fname)
        fname = re.sub(r"_s\d+\.log$", "", fname)
        return f"eval_lr/{fname}"

    # Handle sklearn_baselines
    if parts[0] == "sklearn_baselines":
        method = parts[1]  # lr, svm, xgboost
        return f"sklearn/{method}"

    # Handle CellTypist
    if parts[0] == "CellTypist":
        fname = parts[-1]
        if "record" in fname:
            return "CellTypist (retrained)"
        return "CellTypist"

    # Handle TACCO
    if parts[0] == "TACCO":
        return "TACCO"

    # Handle foundation models with freeze/full
    exp_dir = parts[0]
    fname = parts[-1]
    fname = re.sub(r"^record_", "", fname)
    fname = re.sub(r"^nohup_", "", fname)
    fname = re.sub(r"_s\d+\.log$", "", fname)
    fname = re.sub(r"\.log$", "", fname)

    if fname in ("freeze", "full"):
        return f"{exp_dir} ({fname})"

    # scBrain variants. Legacy experiment directories may still be named mjm*.
    if exp_dir.startswith("mjm"):
        exp_label = {
            "mjm": "scBrainFlat",
            "mjm_1": "scBrain",
            "mjm_2": "scBrainContext",
            "mjm_2_lr": "scBrainContextLR",
            "mjm_attn": "scBrainAttention",
            "mjm_lr": "scBrainLR",
            "mjm_moo": "scBrainMOO",
        }.get(exp_dir, exp_dir.replace("mjm", "scBrain", 1))
        # Clean up common patterns
        fname = re.sub(r"e\d+_d\d+_", "", fname)
        fname = re.sub(r"enc\d+_", "", fname)
        fname = re.sub(r"z\d+_?", "", fname)
        fname = re.sub(r"_+", "_", fname).strip("_")
        if fname:
            return f"{exp_label}/{fname}"
        return exp_label

    if fname:
        return f"{exp_dir}/{fname}"
    return exp_dir


def should_skip_duplicate(filepath, all_files):
    """Skip nohup logs if a corresponding record_ log exists with test metrics.
    Also skip lowercase 'stack' dir if uppercase 'Stack' exists.
    Also skip TACCO/run_v2.log if TACCO/logs/record_s3028.log exists."""
    fname = os.path.basename(filepath)
    dirpath = os.path.dirname(filepath)
    rel = os.path.relpath(filepath, BASE)

    # Skip stack/ if Stack/ exists (lowercase duplicate of uppercase)
    if rel.startswith("stack/"):
        for f in all_files:
            if os.path.relpath(f, BASE).startswith("Stack/"):
                return True

    # Skip TACCO/run_v2.log if TACCO/logs/record exists
    if "TACCO" in rel and ("run_v2.log" in fname or "run.log" in fname):
        for f in all_files:
            frel = os.path.relpath(f, BASE)
            if "TACCO" in frel and "record_" in os.path.basename(f):
                return True

    # If this is a nohup file, check if there's a record_ file in the same experiment
    if fname.startswith("nohup") or fname == "nohup.out":
        # Look for record_ files in the same directory or subdirectories
        parent = dirpath
        # Check one level up too
        for check_dir in [dirpath, os.path.dirname(dirpath)]:
            for f in all_files:
                if f == filepath:
                    continue
                if os.path.dirname(f) == check_dir or os.path.dirname(f).startswith(check_dir):
                    if os.path.basename(f).startswith("record_"):
                        return True
    # Also skip run.log if record_ exists
    if fname == "run.log":
        for f in all_files:
            if f == filepath:
                continue
            if os.path.dirname(f) == dirpath or os.path.dirname(os.path.dirname(f)) == dirpath:
                if os.path.basename(f).startswith("record_"):
                    return True
    return False


def get_metric(results, level, metric, default="--"):
    """Get a metric value from results dict, trying exact key first."""
    if level in results and metric in results[level]:
        return results[level][metric]
    return default


def fmt(val, decimals=4):
    if val == "--":
        return "--"
    return f"{val:.{decimals}f}"


def main():
    all_logs = find_all_log_files()
    print(f"Found {len(all_logs)} log files total\n")

    # Categorize logs
    logs_with_metrics = []
    logs_skipped_dup = []
    logs_still_training = []
    logs_no_metrics = []

    # First pass: find which have metrics
    files_with_test_metrics = set()
    for lf in all_logs:
        results = parse_metrics_from_log(lf)
        if results:
            files_with_test_metrics.add(lf)

    for lf in all_logs:
        if lf in files_with_test_metrics:
            if should_skip_duplicate(lf, list(files_with_test_metrics)):
                logs_skipped_dup.append(lf)
            else:
                logs_with_metrics.append(lf)
        else:
            if check_still_training(lf):
                logs_still_training.append(lf)
            else:
                logs_no_metrics.append(lf)

    # ── Parse all metrics ────────────────────────────────────────────────
    rows = []
    kukanja_rows = []
    for lf in logs_with_metrics:
        results = parse_metrics_from_log(lf)
        if not results:
            continue

        rel = os.path.relpath(lf, BASE)
        name = derive_display_name(lf)

        # Check if this is a kukanja log (uses ANNOTATION_* or LEVEL* keys)
        is_kukanja = any(k.startswith("ANNOTATION_") or k.startswith("LEVEL") for k in results)

        if is_kukanja:
            # Map kukanja levels to standard names
            # ANNOTATION_* format
            if any(k.startswith("ANNOTATION_") for k in results):
                cls_key = "ANNOTATION_BROAD"
                sub_key = "ANNOTATION_MEDIUM"
                sup_key = "ANNOTATION_HIGH"
            else:
                # LEVEL* format
                cls_key = "LEVEL0"
                sub_key = "LEVEL1_5"
                sup_key = "LEVEL3"

            sup_f1 = get_metric(results, sup_key, "macro_f1", 0)
            diverged = isinstance(sup_f1, (int, float)) and sup_f1 < 0.001

            # sub_da_key depends on format
            if any(k.startswith("ANNOTATION_") for k in results):
                sub_da_key = "ANNOTATION_MEDIUM_DA"
            else:
                sub_da_key = "LEVEL2"

            row = {
                "path": rel,
                "name": name,
                "cls_f1": get_metric(results, cls_key, "macro_f1"),
                "sub_f1": get_metric(results, sub_key, "macro_f1"),
                "sup_f1": get_metric(results, sup_key, "macro_f1"),
                "sup_p": get_metric(results, sup_key, "macro_p"),
                "sup_r": get_metric(results, sup_key, "macro_r"),
                "sup_micro_f1": get_metric(results, sup_key, "micro_f1"),
                "sup_auc_macro": get_metric(results, sup_key, "auc_macro"),
                "sup_auc_micro": get_metric(results, sup_key, "auc_micro"),
                "cls_auc_macro": get_metric(results, cls_key, "auc_macro"),
                "sub_auc_macro": get_metric(results, sub_key, "auc_macro"),
                "sub_da_f1": get_metric(results, sub_da_key, "macro_f1"),
                "diverged": diverged,
                "is_kukanja": True,
            }
            kukanja_rows.append(row)
            continue

        # Determine which prefix set to use
        prefixes_found = set()
        for key in results:
            prefix, _ = normalize_level(key)
            prefixes_found.add(prefix)

        # For TACCO, we have PROJECTION_ and OT_ variants - create separate rows
        if "PROJECTION" in prefixes_found or "OT" in prefixes_found:
            for pfx in sorted(prefixes_found):
                if pfx == "":
                    continue  # skip bare keys if prefixed ones exist
                pfx_name = f"{name} ({pfx.lower()})" if pfx else name
                cls_key = f"{pfx}_CLASS" if pfx else "CLASS"
                sub_key = f"{pfx}_SUBCLASS" if pfx else "SUBCLASS"
                sup_key = f"{pfx}_SUPERTYPE" if pfx else "SUPERTYPE"

                sup_f1 = get_metric(results, sup_key, "macro_f1", 0)
                diverged = isinstance(sup_f1, (int, float)) and sup_f1 < 0.001

                rows.append({
                    "path": rel,
                    "name": pfx_name,
                    "cls_f1": get_metric(results, cls_key, "macro_f1"),
                    "sub_f1": get_metric(results, sub_key, "macro_f1"),
                    "sup_f1": get_metric(results, sup_key, "macro_f1"),
                    "sup_p": get_metric(results, sup_key, "macro_p"),
                    "sup_r": get_metric(results, sup_key, "macro_r"),
                    "sup_micro_f1": get_metric(results, sup_key, "micro_f1"),
                    "sup_auc_macro": get_metric(results, sup_key, "auc_macro"),
                    "sup_auc_micro": get_metric(results, sup_key, "auc_micro"),
                    "cls_auc_macro": get_metric(results, cls_key, "auc_macro"),
                    "sub_auc_macro": get_metric(results, sub_key, "auc_macro"),
                    "diverged": diverged,
                })
        else:
            # Standard or RETRAINED_ prefix
            if "RETRAINED" in prefixes_found:
                cls_key, sub_key, sup_key = "RETRAINED_CLASS", "RETRAINED_SUBCLASS", "RETRAINED_SUPERTYPE"
            else:
                cls_key, sub_key, sup_key = "CLASS", "SUBCLASS", "SUPERTYPE"

            sup_f1 = get_metric(results, sup_key, "macro_f1", 0)
            diverged = isinstance(sup_f1, (int, float)) and sup_f1 < 0.001

            rows.append({
                "path": rel,
                "name": name,
                "cls_f1": get_metric(results, cls_key, "macro_f1"),
                "sub_f1": get_metric(results, sub_key, "macro_f1"),
                "sup_f1": get_metric(results, sup_key, "macro_f1"),
                "sup_p": get_metric(results, sup_key, "macro_p"),
                "sup_r": get_metric(results, sup_key, "macro_r"),
                "sup_micro_f1": get_metric(results, sup_key, "micro_f1"),
                "sup_auc_macro": get_metric(results, sup_key, "auc_macro"),
                "sup_auc_micro": get_metric(results, sup_key, "auc_micro"),
                "cls_auc_macro": get_metric(results, cls_key, "auc_macro"),
                "sub_auc_macro": get_metric(results, sub_key, "auc_macro"),
                "diverged": diverged,
            })

    # Sort by SUPERTYPE Macro F1 descending
    def sort_key(r):
        v = r["sup_f1"]
        return v if isinstance(v, float) else -1
    rows.sort(key=sort_key, reverse=True)

    # ── Print main table ─────────────────────────────────────────────────
    sep = "=" * 220
    print(sep)
    print(f"{'#':>3}  {'Display Name':<45} {'CLS F1':>7} {'SUB F1':>7} {'SUP F1':>7} {'SUP P':>7} {'SUP R':>7} {'SUP uF1':>7} {'SUP AUCm':>8} {'SUP AUCu':>8} {'CLS AUCm':>8} {'SUB AUCm':>8}  {'Status':<10}")
    print(sep)

    for i, r in enumerate(rows, 1):
        status = "DIVERGED" if r["diverged"] else ""
        print(f"{i:>3}  {r['name']:<45} {fmt(r['cls_f1']):>7} {fmt(r['sub_f1']):>7} {fmt(r['sup_f1']):>7} {fmt(r['sup_p']):>7} {fmt(r['sup_r']):>7} {fmt(r['sup_micro_f1']):>7} {fmt(r['sup_auc_macro']):>8} {fmt(r['sup_auc_micro']):>8} {fmt(r['cls_auc_macro']):>8} {fmt(r['sub_auc_macro']):>8}  {status:<10}")

    print(sep)
    print(f"\nTotal experiments with test metrics: {len(rows)}")
    print(f"Diverged: {sum(1 for r in rows if r['diverged'])}")

    # ── Print detailed paths ─────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("LOG FILE PATHS (for reference):")
    print(f"{'='*100}")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>3}. {r['name']:<45} -> {r['path']}")

    # ── Skipped duplicates ───────────────────────────────────────────────
    if logs_skipped_dup:
        print(f"\n{'='*100}")
        print(f"SKIPPED DUPLICATE LOGS ({len(logs_skipped_dup)}):")
        print(f"{'='*100}")
        for lf in logs_skipped_dup:
            print(f"  {os.path.relpath(lf, BASE)}")

    # ── Still training ───────────────────────────────────────────────────
    if logs_still_training:
        print(f"\n{'='*100}")
        print(f"STILL TRAINING / NO TEST RESULTS ({len(logs_still_training)}):")
        print(f"{'='*100}")
        for lf in sorted(logs_still_training):
            rel = os.path.relpath(lf, BASE)
            # Get last epoch
            with open(lf, "r", errors="replace") as f:
                content = f.read()
            epochs = re.findall(r"Epoch \[(\d+)/(\d+)\]", content)
            if epochs:
                last_ep, max_ep = epochs[-1]
                print(f"  {rel:<70} (epoch {last_ep}/{max_ep})")
            else:
                print(f"  {rel}")

    # ── No metrics at all ────────────────────────────────────────────────
    if logs_no_metrics:
        print(f"\n{'='*100}")
        print(f"LOGS WITH NO TEST METRICS AND NOT TRAINING ({len(logs_no_metrics)}):")
        print(f"{'='*100}")
        for lf in sorted(logs_no_metrics):
            print(f"  {os.path.relpath(lf, BASE)}")

    # ── Check directories with no completed results ──────────────────────
    print(f"\n{'='*100}")
    print("EXPERIMENT DIRECTORY STATUS SUMMARY:")
    print(f"{'='*100}")
    exp_dirs = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])
    dirs_with_results = set()
    for r in rows:
        # Get top-level dir from path
        top = r["path"].split("/")[0]
        dirs_with_results.add(top)
    for lf in logs_skipped_dup:
        top = os.path.relpath(lf, BASE).split("/")[0]
        dirs_with_results.add(top)

    # Also count kukanja in dirs_with_results
    for r in kukanja_rows:
        top = r["path"].split("/")[0]
        dirs_with_results.add(top)

    for d in exp_dirs:
        if d in dirs_with_results:
            # Count how many results
            count = sum(1 for r in rows if r["path"].split("/")[0] == d)
            print(f"  [OK]       {d:<30} ({count} result(s))")
        else:
            # Check if still training
            training = [lf for lf in logs_still_training if os.path.relpath(lf, BASE).split("/")[0] == d]
            if training:
                print(f"  [TRAINING] {d:<30} (still training)")
            else:
                print(f"  [NONE]     {d:<30} (no test results)")

    # ── Batch integration results ────────────────────────────────────────
    scib_csv = BASE / "batch_integration" / "SEA_AD_MTP_ST" / "scib_fast_results.csv"
    if scib_csv.exists():
        print(f"\n{'='*100}")
        print("BATCH INTEGRATION (scIB) RESULTS:")
        print(f"{'='*100}")
        import csv
        with open(scib_csv) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"  {'Method':<35} {'batch':>7} {'bio':>7} {'overall':>7} {'ASW_b':>7} {'graph':>7} {'PCR_b':>7} {'ASW_l':>7} {'NMI':>7} {'ARI':>7}")
            print(f"  {'-'*100}")
            for row in reader:
                name = row.get("", row.get("Unnamed: 0", "?"))
                print(f"  {name:<35} {float(row.get('batch_score',0)):.4f} {float(row.get('bio_score',0)):.4f} {float(row.get('overall',0)):.4f} {float(row.get('ASW_batch',0)):.4f} {float(row.get('graph_conn',0)):.4f} {float(row.get('PCR_batch',0)):.4f} {float(row.get('ASW_label',0)):.4f} {float(row.get('NMI',0)):.4f} {float(row.get('ARI',0)):.4f}")

    # ── Kukanja summary ──────────────────────────────────────────────────
    # kukanja_rows is already populated from the main loop
    if kukanja_rows:
        print(f"\n{'='*160}")
        print("KUKANJA TRANSFER LEARNING RESULTS (different label hierarchy: BROAD/MEDIUM/MEDIUM_DA/HIGH):")
        print(f"{'='*160}")
        for dataset in ["Kukanja_EAE", "Kukanja_MS"]:
            ds_rows = [r for r in kukanja_rows if dataset in r["name"]]
            if ds_rows:
                ds_rows.sort(key=lambda r: r["sup_f1"] if isinstance(r["sup_f1"], float) else -1, reverse=True)
                print(f"\n  --- {dataset} (BROAD=class, MEDIUM=subclass, HIGH=supertype) ---")
                print(f"  {'#':>3}  {'Name':<55} {'BROAD F1':>8} {'MED F1':>8} {'HIGH F1':>8} {'HIGH P':>8} {'HIGH R':>8} {'HIGH uF1':>8} {'HIGH AUCm':>9} {'CLS AUCm':>9} {'SUB AUCm':>9}")
                for i, r in enumerate(ds_rows, 1):
                    print(f"  {i:>3}  {r['name']:<55} {fmt(r['cls_f1']):>8} {fmt(r['sub_f1']):>8} {fmt(r['sup_f1']):>8} {fmt(r['sup_p']):>8} {fmt(r['sup_r']):>8} {fmt(r['sup_micro_f1']):>8} {fmt(r['sup_auc_macro']):>9} {fmt(r['cls_auc_macro']):>9} {fmt(r['sub_auc_macro']):>9}")

main()
