import argparse
import copy
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from first_layer_stability import (
    BASELINE_PRESETS,
    TARGET_LAYER1_PRESETS,
    TARGET_LAYER2_PRESETS,
    TrainConfig,
    append_experiment_cells_to_notebook,
    clone_training_state,
    compute_stability,
    copy_dataset,
    ensure_dir,
    final_metrics_row,
    initialize_training_state,
    load_dataset_bundle,
    parse_generator_commands,
    plot_training_curves,
    run_generation,
    save_dataset_inventory,
    save_stage_summary,
    train_stage,
    write_notes_file,
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def build_experiment_root(base_dir: Path, stamp: Optional[str] = None) -> Path:
    stamp = stamp or now_stamp()
    root = base_dir / f"first_layer_stability_{stamp}"
    ensure_dir(root)
    for rel in (
        "datasets",
        "runs/baselines",
        "runs/three_layer_control",
        "runs/shift_layer1",
        "runs/shift_layer2",
        "plots/summary",
        "tables",
        "notes",
    ):
        ensure_dir(root / rel)
    return root


def select_specs(specs, presets: Sequence[str]):
    preset_set = set(presets)
    return [spec for spec in specs if spec.preset in preset_set]


def maybe_copy_dataset_to_experiment(spec, experiment_root: Path) -> Path:
    return copy_dataset(spec.output_path, experiment_root / "datasets")


def write_final_correlation_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    corr_cols = [col for col in metrics_df.columns if col.startswith("corr_")]
    final_row = metrics_df.iloc[-1]
    rows = [{"latent": col.replace("corr_", ""), "corr": final_row[col]} for col in corr_cols]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def make_baseline_summary_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    summary_df = summary_df.copy()
    summary_df["preset_label"] = summary_df["preset"].str.replace("_", "\n")

    axes[0].bar(summary_df["preset_label"], summary_df["best_first_layer_corr"], color="tab:blue")
    axes[0].set_title("Best First-Layer Correlation")
    axes[0].set_ylabel("abs corr")

    axes[1].bar(summary_df["preset_label"], summary_df["best_non_first_layer_corr"], color="tab:gray")
    axes[1].set_title("Best Non-First-Layer Correlation")
    axes[1].set_ylabel("abs corr")

    axes[2].bar(summary_df["preset_label"], summary_df["selectivity_margin"], color="tab:green")
    axes[2].set_title("Selectivity Margin")
    axes[2].set_ylabel("margin")
    axes[2].tick_params(axis="x", labelrotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_shift_family_plot(summary_df: pd.DataFrame, title: str, output_path: Path) -> None:
    summary_df = summary_df.sort_values("variance")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

    axes[0].plot(summary_df["variance"], summary_df["selectivity_margin"], marker="o")
    axes[0].set_title("Final Selectivity Margin")
    axes[0].set_xlabel("variance")
    axes[0].set_ylabel("margin")

    axes[1].plot(summary_df["variance"], summary_df["cmi"], marker="o", color="tab:orange")
    axes[1].set_title("Final CMI")
    axes[1].set_xlabel("variance")
    axes[1].set_ylabel("cmi")

    axes[2].plot(summary_df["variance"], summary_df["best_first_layer_corr"], marker="o", color="tab:green")
    axes[2].set_title("Best First-Layer Correlation")
    axes[2].set_xlabel("variance")
    axes[2].set_ylabel("abs corr")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_est_std_plot(summary_df: pd.DataFrame, title: str, output_path: Path) -> None:
    summary_df = summary_df.sort_values("variance")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(summary_df["variance"], summary_df["std_est_z1t0"], marker="o")
    ax.set_title(title)
    ax.set_xlabel("variance")
    ax.set_ylabel("std(est_z1t0)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def clone_cfg(train_cfg: TrainConfig, **updates) -> TrainConfig:
    data = train_cfg.to_json_dict()
    data.update(updates)
    data["snapshot_epochs"] = tuple(data["snapshot_epochs"])
    return TrainConfig(**data)


def run_fresh_experiment(
    spec,
    experiment_root: Path,
    train_cfg: TrainConfig,
    device: str,
    run_group: str,
) -> Dict[str, object]:
    run_dir = experiment_root / "runs" / run_group / spec.preset
    ensure_dir(run_dir)
    dataset_bundle = load_dataset_bundle(
        npz_path=spec.output_path,
        dataset_name=spec.preset,
        device=device,
        n_layers=spec.n_layers,
        n_vars_per_layer=spec.n_vars_per_layer,
    )
    state = initialize_training_state(dataset=dataset_bundle, train_cfg=train_cfg, device=device)
    metrics_df = train_stage(
        state=state,
        dataset=dataset_bundle,
        stage_id="stage_a",
        output_dir=run_dir,
        note="fresh training",
    )
    write_final_correlation_table(metrics_df, run_dir / "final_correlations.csv")
    final_row = metrics_df.iloc[-1].to_dict()
    stable, checks = compute_stability(final_row)
    return {
        "preset": spec.preset,
        "dataset_path": str(spec.output_path),
        "stable": stable,
        "checks": json.dumps(checks),
        **final_row,
    }


def run_stage_a_control(
    spec,
    experiment_root: Path,
    train_cfg: TrainConfig,
    device: str,
) -> Tuple[Dict[str, object], object, object, pd.DataFrame]:
    run_dir = experiment_root / "runs" / "three_layer_control"
    ensure_dir(run_dir)

    ladder = [
        ("default", train_cfg),
        ("lambda_cmi_1e-3", clone_cfg(train_cfg, lambda_cmi=1e-3)),
        ("lambda_cmi_1e-4", clone_cfg(train_cfg, lambda_cmi=1e-4)),
        ("ema_0.99", clone_cfg(train_cfg, ema_m=0.99)),
    ]
    attempts = []
    final_bundle = None
    final_state = None
    final_metrics_df = None
    final_label = None
    for label, cfg in ladder:
        attempt_dir = run_dir if label == "default" else run_dir / f"rerun_{label.replace('.', '_')}"
        ensure_dir(attempt_dir)
        dataset_bundle = load_dataset_bundle(
            npz_path=spec.output_path,
            dataset_name=spec.preset,
            device=device,
            n_layers=spec.n_layers,
            n_vars_per_layer=spec.n_vars_per_layer,
        )
        state = initialize_training_state(dataset=dataset_bundle, train_cfg=cfg, device=device)
        metrics_df = train_stage(state=state, dataset=dataset_bundle, stage_id="stage_a", output_dir=attempt_dir, note=f"control run with {label}")
        final_row = metrics_df.iloc[-1].to_dict()
        stable, checks = compute_stability(final_row)
        attempts.append({"attempt": label, "stable": stable, "checks": checks, "final_metrics": final_row})
        if stable:
            final_bundle = dataset_bundle
            final_state = state
            final_metrics_df = metrics_df
            final_label = label
            break
        if final_bundle is None:
            final_bundle = dataset_bundle
            final_state = state
            final_metrics_df = metrics_df
            final_label = label

    save_stage_summary({"attempts": attempts, "selected": final_label}, run_dir / "fallback_attempts.json")
    (run_dir / "selected_attempt.json").write_text(json.dumps({"selected": final_label}, indent=2), encoding="utf-8")
    write_final_correlation_table(final_metrics_df, run_dir / "final_correlations.csv")
    control_row = {
        "preset": spec.preset,
        "dataset_path": str(spec.output_path),
        "fallback_label": final_label,
        **final_metrics_df.iloc[-1].to_dict(),
    }
    return control_row, final_bundle, final_state, final_metrics_df


def run_continuation_branch(
    base_state,
    stage_a_metrics_df: pd.DataFrame,
    shift_spec,
    experiment_root: Path,
    device: str,
    family_dir: str,
    variance: float,
    family_label: str,
) -> Dict[str, object]:
    branch_dir = experiment_root / "runs" / family_dir / f"variance_{variance:0.2f}"
    ensure_dir(branch_dir)
    base_bundle = load_dataset_bundle(
        npz_path=shift_spec.output_path,
        dataset_name=shift_spec.preset,
        device=device,
        n_layers=shift_spec.n_layers,
        n_vars_per_layer=shift_spec.n_vars_per_layer,
    )

    ladder = [
        ("default", {"lambda_cmi": base_state.train_cfg.lambda_cmi, "ema_m": base_state.train_cfg.ema_m}),
        ("lambda_cmi_1e-3", {"lambda_cmi": 1e-3, "ema_m": base_state.train_cfg.ema_m}),
        ("lambda_cmi_1e-4", {"lambda_cmi": 1e-4, "ema_m": base_state.train_cfg.ema_m}),
        ("ema_0.99", {"lambda_cmi": base_state.train_cfg.lambda_cmi, "ema_m": 0.99}),
    ]

    attempts = []
    selected_df = None
    selected_state = None
    selected_label = None

    for label, updates in ladder:
        state = clone_training_state(base_state)
        state.train_cfg = clone_cfg(state.train_cfg, **updates)
        attempt_dir = branch_dir if label == "default" else branch_dir / f"rerun_{label.replace('.', '_')}"
        metrics_df = train_stage(
            state=state,
            dataset=base_bundle,
            stage_id="stage_b",
            output_dir=attempt_dir / "stage_b",
            note=f"{family_label} shift continuation with {label}",
        )
        final_row = metrics_df.iloc[-1].to_dict()
        stable, checks = compute_stability(final_row)
        attempts.append({"attempt": label, "stable": stable, "checks": checks, "final_metrics": final_row})
        if stable:
            selected_df = metrics_df
            selected_state = state
            selected_label = label
            branch_root = attempt_dir
            break
        if selected_df is None:
            selected_df = metrics_df
            selected_state = state
            selected_label = label
            branch_root = attempt_dir

    stage_a_clean = stage_a_metrics_df.copy()
    stage_b_clean = selected_df.copy()
    combined = pd.concat([stage_a_clean, stage_b_clean], ignore_index=True)
    combined.to_csv(branch_root / "combined_metrics.csv", index=False)
    plot_training_curves(
        metrics_df=combined[combined["is_pre_stage"] == 0],
        dataset=base_bundle,
        output_path=branch_root / "combined_training_curves.png",
        stage_boundary=int(stage_a_metrics_df["epoch"].max()),
    )
    write_final_correlation_table(selected_df, branch_root / "final_correlations.csv")
    save_stage_summary({"attempts": attempts, "selected": selected_label}, branch_root / "fallback_attempts.json")

    final_row = selected_df.iloc[-1].to_dict()
    stable, checks = compute_stability(final_row)
    (branch_dir / "selected_attempt.json").write_text(json.dumps({"selected": selected_label}, indent=2), encoding="utf-8")
    return {
        "preset": shift_spec.preset,
        "variance": variance,
        "dataset_path": str(shift_spec.output_path),
        "family": family_label,
        "fallback_label": selected_label,
        "stable": stable,
        "checks": json.dumps(checks),
        **final_row,
    }


def create_report_notebook(experiment_root: Path) -> Path:
    report_path = experiment_root / "report_first_layer_stability.ipynb"
    root_repr = repr(str(experiment_root))
    cells = [
        new_markdown_cell("# First-Layer Synthetic Stability Report"),
        new_markdown_cell(
            "## Objective and Setup\n"
            "This report summarizes the baseline sweep and the targeted 3-layer distribution-shift study using the current scalar first-layer estimator."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            f"ROOT = Path({root_repr})\n"
            "baseline = pd.read_csv(ROOT / 'tables' / 'baseline_summary.csv')\n"
            "control = pd.read_csv(ROOT / 'tables' / 'three_layer_control_summary.csv')\n"
            "shift_l1 = pd.read_csv(ROOT / 'tables' / 'shift_layer1_summary.csv')\n"
            "shift_l2 = pd.read_csv(ROOT / 'tables' / 'shift_layer2_summary.csv')\n"
            "inventory = pd.read_csv(ROOT / 'tables' / 'dataset_inventory.csv')\n"
            "baseline"
        ),
        new_markdown_cell("## Dataset Inventory"),
        new_code_cell("inventory"),
        new_markdown_cell("## Baseline Preset Results\n\n![](plots/summary/baseline_summary.png)"),
        new_code_cell("baseline"),
        new_markdown_cell("## Targeted 3-Layer Control Results"),
        new_code_cell("control"),
        new_markdown_cell("## Second-Layer Shift Continuation Results\n\n![](plots/summary/shift_layer2_summary.png)\n\n![](plots/summary/shift_layer2_std.png)"),
        new_code_cell("shift_l2"),
        new_markdown_cell("## First-Layer Shift Continuation Results\n\n![](plots/summary/shift_layer1_summary.png)\n\n![](plots/summary/shift_layer1_std.png)"),
        new_code_cell("shift_l1"),
        new_markdown_cell("## Stability Notes and Failure Cases"),
        new_code_cell("(ROOT / 'notes' / 'run_notes.md').read_text()"),
        new_markdown_cell("## Conclusions"),
        new_code_cell("(ROOT / 'report_summary.md').read_text()"),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.write(nb, report_path)
    return report_path


def export_report(report_path: Path) -> None:
    import subprocess

    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(report_path),
        ],
        cwd=report_path.parent,
        check=True,
    )
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            str(report_path),
            "--output",
            "report_first_layer_stability.html",
        ],
        cwd=report_path.parent,
        check=True,
    )


def write_report_summary(
    experiment_root: Path,
    baseline_df: pd.DataFrame,
    control_df: pd.DataFrame,
    shift_l1_df: pd.DataFrame,
    shift_l2_df: pd.DataFrame,
    fallback_rows: List[Dict[str, object]],
) -> None:
    stable_l2 = bool((shift_l2_df["selectivity_margin"] > 0.0).all())
    stable_l1 = bool((shift_l1_df["selectivity_margin"] > 0.0).all())
    layer1_std_monotonic = shift_l1_df.sort_values("variance")["std_est_z1t0"].is_monotonic_increasing

    lines = [
        f"Experiment root: {experiment_root}",
        f"Experiment timestamp: {experiment_root.name.split('first_layer_stability_')[-1]}",
        f"Environment: Python / torch on {device_name()}",
        "",
        "Datasets:",
    ]
    inventory = pd.read_csv(experiment_root / "tables" / "dataset_inventory.csv")
    for _, row in inventory.iterrows():
        lines.append(f"- {row['preset']}: {row['output_path']}")

    lines.extend(["", "Fallback reruns used:"])
    if fallback_rows:
        for row in fallback_rows:
            lines.append(f"- {row['family']} variance={row['variance']}: {row['fallback_label']}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Top-level findings:",
            f"- First-layer recovery stable under second-layer shift: {'yes' if stable_l2 else 'no'}",
            f"- First-layer recovery stable under first-layer shift: {'yes' if stable_l1 else 'no'}",
            f"- Increasing top-layer variance increases observed top-layer recovery variance: {'yes' if layer1_std_monotonic else 'no'}",
            f"- Three-layer control final best first-layer correlation: {control_df.iloc[0]['best_first_layer_corr']:.4f}",
            f"- Three-layer control final selectivity margin: {control_df.iloc[0]['selectivity_margin']:.4f}",
        ]
    )
    (experiment_root / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-layer synthetic stability experiments and produce a report.")
    parser.add_argument("--experiment-root", type=Path, default=None, help="Optional output directory. Defaults to timestamped experiments path.")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate datasets even if they already exist.")
    parser.add_argument("--n-epochs", type=int, default=1000, help="Epochs per stage.")
    parser.add_argument("--n-q-epochs", type=int, default=20, help="q-updates per epoch.")
    parser.add_argument("--n-f-epochs", type=int, default=3, help="f-updates per epoch.")
    parser.add_argument("--warmup-epochs", type=int, default=50, help="Warmup epochs.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    experiments_root = project_root / "experiments"
    ensure_dir(experiments_root)
    experiment_root = args.experiment_root or build_experiment_root(experiments_root)
    ensure_dir(experiment_root)
    for rel in (
        "datasets",
        "runs/baselines",
        "runs/three_layer_control",
        "runs/shift_layer1",
        "runs/shift_layer2",
        "plots/summary",
        "tables",
        "notes",
    ):
        ensure_dir(experiment_root / rel)

    script_path = project_root / "data" / "run_generator_multiple.sh"
    specs = parse_generator_commands(script_path)
    baseline_specs = select_specs(specs, BASELINE_PRESETS)
    target_presets = ["three_layer_l_nonlinear_base", *TARGET_LAYER1_PRESETS.values(), *TARGET_LAYER2_PRESETS.values()]
    target_specs = select_specs(specs, target_presets)
    used_specs = baseline_specs + [spec for spec in target_specs if spec.preset not in {s.preset for s in baseline_specs}]

    run_generation(specs=used_specs, script_dir=script_path.parent, force=args.force_generate)
    for spec in used_specs:
        maybe_copy_dataset_to_experiment(spec, experiment_root)

    save_dataset_inventory(used_specs, experiment_root / "tables" / "dataset_inventory.csv")
    append_experiment_cells_to_notebook(
        notebook_path=project_root / "estimate_first_layer.ipynb",
        experiment_root_expr=f"Path({str(experiment_root)!r})",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_cfg = TrainConfig(
        n_epochs=args.n_epochs,
        n_q_epochs=args.n_q_epochs,
        n_f_epochs=args.n_f_epochs,
        warmup_epochs=args.warmup_epochs,
    )

    baseline_rows = []
    for spec in baseline_specs:
        baseline_rows.append(run_fresh_experiment(spec=spec, experiment_root=experiment_root, train_cfg=default_cfg, device=device, run_group="baselines"))

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(experiment_root / "tables" / "baseline_summary.csv", index=False)
    make_baseline_summary_plot(baseline_df, experiment_root / "plots" / "summary" / "baseline_summary.png")

    control_spec = next(spec for spec in target_specs if spec.preset == "three_layer_l_nonlinear_base")
    control_row, control_bundle, control_state, control_metrics_df = run_stage_a_control(
        spec=control_spec,
        experiment_root=experiment_root,
        train_cfg=default_cfg,
        device=device,
    )
    control_df = pd.DataFrame([control_row])
    control_df.to_csv(experiment_root / "tables" / "three_layer_control_summary.csv", index=False)

    shift_layer1_rows: List[Dict[str, object]] = []
    shift_layer2_rows: List[Dict[str, object]] = []
    fallback_rows: List[Dict[str, object]] = []

    for variance, preset in TARGET_LAYER2_PRESETS.items():
        shift_spec = next(spec for spec in target_specs if spec.preset == preset)
        row = run_continuation_branch(
            base_state=control_state,
            stage_a_metrics_df=control_metrics_df,
            shift_spec=shift_spec,
            experiment_root=experiment_root,
            device=device,
            family_dir="shift_layer2",
            variance=variance,
            family_label="layer2_shift",
        )
        shift_layer2_rows.append(row)
        if row["fallback_label"] != "default":
            fallback_rows.append({"family": "layer2_shift", "variance": variance, "fallback_label": row["fallback_label"]})

    for variance, preset in TARGET_LAYER1_PRESETS.items():
        shift_spec = next(spec for spec in target_specs if spec.preset == preset)
        row = run_continuation_branch(
            base_state=control_state,
            stage_a_metrics_df=control_metrics_df,
            shift_spec=shift_spec,
            experiment_root=experiment_root,
            device=device,
            family_dir="shift_layer1",
            variance=variance,
            family_label="layer1_shift",
        )
        shift_layer1_rows.append(row)
        if row["fallback_label"] != "default":
            fallback_rows.append({"family": "layer1_shift", "variance": variance, "fallback_label": row["fallback_label"]})

    shift_l1_df = pd.DataFrame(shift_layer1_rows).sort_values("variance")
    shift_l2_df = pd.DataFrame(shift_layer2_rows).sort_values("variance")
    shift_l1_df.to_csv(experiment_root / "tables" / "shift_layer1_summary.csv", index=False)
    shift_l2_df.to_csv(experiment_root / "tables" / "shift_layer2_summary.csv", index=False)

    make_shift_family_plot(shift_l1_df, "First-Layer Shift Summary", experiment_root / "plots" / "summary" / "shift_layer1_summary.png")
    make_shift_family_plot(shift_l2_df, "Second-Layer Shift Summary", experiment_root / "plots" / "summary" / "shift_layer2_summary.png")
    make_est_std_plot(shift_l1_df, "Estimated Latent Std vs Layer-1 Variance", experiment_root / "plots" / "summary" / "shift_layer1_std.png")
    make_est_std_plot(shift_l2_df, "Estimated Latent Std vs Layer-2 Variance", experiment_root / "plots" / "summary" / "shift_layer2_std.png")

    notes_lines = [
        "Run notes",
        f"- Device: {device_name()}",
        "- Scalar first-layer estimator preserved from estimate_first_layer.py / notebook.",
        "- Reconstruction term used only during warmup.",
        "- lr_f and lr_pred kept <= lr_q.",
        "- n_q_epochs kept > n_f_epochs.",
        f"- Fallback reruns used: {len(fallback_rows)}",
    ]
    if fallback_rows:
        for row in fallback_rows:
            notes_lines.append(f"- {row['family']} variance={row['variance']}: {row['fallback_label']}")
    write_notes_file(experiment_root / "notes" / "run_notes.md", notes_lines)
    (experiment_root / "notes" / "fallback_rows.json").write_text(json.dumps(fallback_rows, indent=2), encoding="utf-8")

    write_report_summary(
        experiment_root=experiment_root,
        baseline_df=baseline_df,
        control_df=control_df,
        shift_l1_df=shift_l1_df,
        shift_l2_df=shift_l2_df,
        fallback_rows=fallback_rows,
    )
    report_path = create_report_notebook(experiment_root)
    export_report(report_path)

    manifest = {
        "experiment_root": str(experiment_root),
        "device": device_name(),
        "baseline_runs": len(baseline_df),
        "shift_layer1_runs": len(shift_l1_df),
        "shift_layer2_runs": len(shift_l2_df),
    }
    (experiment_root / "tables" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
