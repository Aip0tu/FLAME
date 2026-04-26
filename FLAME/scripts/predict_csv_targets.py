import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FLAME import flsf_predict


DEFAULT_TARGETS = ["abs", "emi", "plqy", "e"]


def resolve_repo_root() -> Path:
    return REPO_ROOT


def normalize_pairs(smiles_series: pd.Series, solvents: pd.Series) -> list[list[str]]:
    pairs: list[list[str]] = []
    for smiles, solvent in zip(smiles_series.fillna(""), solvents.fillna("O")):
        pairs.append([str(smiles).strip(), str(solvent).strip() or "O"])
    return pairs


def predict_one_target(
    target: str,
    smiles_pairs: list[list[str]],
    model_root: Path,
    model_prefix: str,
) -> list[object]:
    model_path = model_root / f"{model_prefix}_{target}"
    preds = flsf_predict(model_path=str(model_path), smiles=smiles_pairs)
    return np.asarray(preds, dtype=object).reshape(-1).tolist()


def main() -> None:
    repo_root = resolve_repo_root()
    default_input = repo_root / "data" / "FluoDB" / "emi_train_validation_results.csv"

    parser = argparse.ArgumentParser(
        description="Predict abs/emi/plqy/e for SMILES in a CSV and append the predictions as new columns."
    )
    parser.add_argument(
        "--input_csv",
        default=str(default_input),
        help="Input CSV path. Defaults to data/FluoDB/flame_rl_em1000_2.csv.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Output CSV path. Defaults to <input_stem>_with_preds.csv next to the input file.",
    )
    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="Column name containing fluorophore SMILES.",
    )
    parser.add_argument(
        "--solvent_col",
        default="",
        help="Optional solvent SMILES column. If omitted or missing, --default_solvent is used.",
    )
    parser.add_argument(
        "--default_solvent",
        default="O",
        help="Fallback solvent SMILES when no solvent column is present. Defaults to water (O).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Targets to predict. Defaults to: abs emi plqy e",
    )
    parser.add_argument(
        "--model_root",
        default=str(repo_root / "model" / "flsf"),
        help="Root directory containing per-target FLSF models such as model/flsf/FluoDB_emi.",
    )
    parser.add_argument(
        "--model_prefix",
        default="FluoDB",
        help="Model prefix used in per-target folder names, e.g. FluoDB_emi.",
    )
    parser.add_argument(
        "--suffix",
        default="_pred",
        help="Suffix appended to each predicted target column name.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else input_csv.with_name(f"{input_csv.stem}_with_preds.csv")
    )
    model_root = Path(args.model_root).resolve()

    df = pd.read_csv(input_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {input_csv}")

    if args.solvent_col and args.solvent_col in df.columns:
        solvents = df[args.solvent_col]
    else:
        solvents = pd.Series([args.default_solvent] * len(df), index=df.index)

    smiles_pairs = normalize_pairs(df[args.smiles_col], solvents)

    for target in args.targets:
        pred_col = f"{target}{args.suffix}"
        df[pred_col] = predict_one_target(
            target=target,
            smiles_pairs=smiles_pairs,
            model_root=model_root,
            model_prefix=args.model_prefix,
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved predictions to {output_csv}")
    print(f"Appended columns: {', '.join(f'{target}{args.suffix}' for target in args.targets)}")


if __name__ == "__main__":
    main()
