#!/usr/bin/env python
"""
Collect all Kukanja experiment results into a unified markdown report.
Parses log files from experiments/kukanja/{Kukanja_MS,Kukanja_EAE}/.
"""
import os, re, sys

BASE = '.'
OUT_ROOT = os.environ.get('KUKANJA_OUT_ROOT', os.path.join(BASE, 'experiments', 'kukanja'))

# Model definitions: {display_name: log_path}
# Will be populated dynamically by scanning log directories

def find_log_files():
    """Scan for all Kukanja experiment log files."""
    models = {}
    roots = [os.path.join(BASE, 'experiments', 'kukanja')]
    if OUT_ROOT not in roots:
        roots.insert(0, OUT_ROOT)
    for root in roots:
        for dataset in ['Kukanja_MS', 'Kukanja_EAE']:
            log_dir = os.path.join(root, dataset)
            if not os.path.isdir(log_dir):
                continue
            for f in sorted(os.listdir(log_dir)):
                if f.endswith('.log') and f.startswith('record_'):
                    tag = f.replace('record_', '').replace('.log', '')
                    display = f"{dataset} / {tag}"
                    models.setdefault(display, os.path.join(log_dir, f))
    return models


def parse_metrics(log_path):
    """Parse test classification metrics from a log file."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        text = f.read()

    # Find last occurrence of test metrics section
    sections = list(re.finditer(r'Test Classification Metrics', text))
    if not sections:
        return None

    # Parse from the last test section
    last_pos = sections[-1].start()
    block = text[last_pos:]

    results = {}
    # Pattern: [LEVEL] Macro  - P:0.xxxx R:0.xxxx F1:0.xxxx
    # OR:      [LEVEL] Macro  - Precision: 0.xxxx | Recall: 0.xxxx | F1: 0.xxxx
    for m in re.finditer(
        r'\[(\w+)\]\s+Macro\s+-\s+(?:P(?:recision)?[:=]\s*)([\d.]+)\s*[|,\s]+(?:R(?:ecall)?[:=]\s*)([\d.]+)\s*[|,\s]+(?:F1[:=]\s*)([\d.]+)',
        block
    ):
        level = m.group(1)
        results[level] = {
            'macro_p': float(m.group(2)),
            'macro_r': float(m.group(3)),
            'macro_f1': float(m.group(4)),
        }

    for m in re.finditer(
        r'\[(\w+)\]\s+Micro\s+-\s+(?:P(?:recision)?[:=]\s*)([\d.]+)\s*[|,\s]+(?:R(?:ecall)?[:=]\s*)([\d.]+)\s*[|,\s]+(?:F1[:=]\s*)([\d.]+)',
        block
    ):
        level = m.group(1)
        if level in results:
            results[level]['micro_f1'] = float(m.group(4))

    # AUC-ROC section
    auc_sections = list(re.finditer(r'AUC-ROC Metrics', text))
    if auc_sections:
        auc_block = text[auc_sections[-1].start():]
        for m in re.finditer(
            r'\[(\w+)\]\s+AUC-ROC\s+-\s+Macro[:=]\s*([\d.]+)\s*[|,\s]+Micro[:=]\s*([\d.]+)',
            auc_block
        ):
            level = m.group(1)
            if level in results:
                results[level]['macro_auc'] = float(m.group(2))
                results[level]['micro_auc'] = float(m.group(3))

    return results if results else None


def pretty_model_name(tag):
    """Map legacy run tags to public scBrain display names."""
    replacements = [
        ("mjm1_lr_", "scBrain linear probe "),
        ("mjm_lr_", "scBrainFlat linear probe "),
        ("mjm2c_lr_", "scBrainFiLM linear probe "),
        ("mjm_attn_", "scBrain attention ablation "),
        ("mjm1_", "scBrain "),
        ("mjm_", "scBrainFlat "),
        ("scbrain_", "scBrain "),
    ]
    for old, new in replacements:
        if tag.startswith(old):
            return new + tag[len(old):]
    return tag


def generate_report(models):
    """Generate unified markdown report."""
    lines = [
        "# Kukanja Dataset Experiment Results",
        "",
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    for dataset_tag in ['Kukanja_MS', 'Kukanja_EAE']:
        ds_models = {k: v for k, v in models.items() if k.startswith(dataset_tag)}
        if not ds_models:
            continue

        lines.append(f"## {dataset_tag}")
        lines.append("")

        # Overview table
        lines.append("### Overview")
        lines.append("")

        # Collect all levels from first model
        all_levels = set()
        for name, path in ds_models.items():
            res = parse_metrics(path)
            if res:
                all_levels.update(res.keys())
                break

        sorted_levels = sorted(all_levels)
        if not sorted_levels:
            lines.append("No test results found yet.")
            lines.append("")
            continue

        # Header
        header = "| Model |"
        sep = "|-------|"
        for lv in sorted_levels:
            header += f" {lv} Macro F1 | {lv} Micro F1 |"
            sep += "------------|------------|"
        lines.append(header)
        lines.append(sep)

        # Rows
        for name, path in sorted(ds_models.items()):
            res = parse_metrics(path)
            row = f"| {pretty_model_name(name.split(' / ')[1])} |"
            if res is None:
                row += " N/A |" * (len(sorted_levels) * 2)
            else:
                for lv in sorted_levels:
                    if lv in res:
                        row += f" {res[lv]['macro_f1']:.4f} | {res[lv].get('micro_f1', 0):.4f} |"
                    else:
                        row += " - | - |"
            lines.append(row)

        lines.append("")

        # Detailed per-level metrics
        lines.append("### Detailed Metrics")
        lines.append("")
        for name, path in sorted(ds_models.items()):
            res = parse_metrics(path)
            if res is None:
                continue
            short_name = pretty_model_name(name.split(' / ')[1])
            lines.append(f"#### {short_name}")
            lines.append("")
            lines.append("| Level | Macro P | Macro R | Macro F1 | Micro F1 | Macro AUC | Micro AUC |")
            lines.append("|-------|---------|---------|----------|----------|-----------|-----------|")
            for lv in sorted_levels:
                if lv in res:
                    r = res[lv]
                    lines.append(
                        f"| {lv} | {r['macro_p']:.4f} | {r['macro_r']:.4f} | "
                        f"{r['macro_f1']:.4f} | {r.get('micro_f1', 0):.4f} | "
                        f"{r.get('macro_auc', 0):.4f} | {r.get('micro_auc', 0):.4f} |"
                    )
            lines.append("")

    return "\n".join(lines)


if __name__ == '__main__':
    models = find_log_files()
    if not models:
        print("No log files found. Run experiments first.")
        sys.exit(1)

    print(f"Found {len(models)} log files:")
    for name, path in sorted(models.items()):
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{exists}] {name}")

    report = generate_report(models)
    out_path = os.path.join(BASE, 'experiments', 'kukanja', 'UNIFIED_RESULTS.md')
    with open(out_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")
