import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from FLAME import flsf_predict


def is_valid_smiles(smiles: object) -> bool:
    if pd.isna(smiles):
        return False
    smiles = str(smiles).strip()
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")

    targets = ["abs", "emi", "plqy", "e"]

    os.makedirs(REPO_ROOT / "pred" / "test", exist_ok=True)

    input_file = REPO_ROOT / "data" / "FluoDB" / "emi_train_validation_results.csv"
    output_file = REPO_ROOT / "pred" / "test" / "emi_train_validation_results_1.csv"

    df = pd.read_csv(input_file)

    smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
    if smiles_col not in df.columns:
        raise ValueError(f"Cannot find a SMILES column in {input_file}")

    # DMSO
    df["solvent"] = "CS(C)=O"
    valid_mask = df[smiles_col].apply(is_valid_smiles)
    valid_df = df.loc[valid_mask].copy()
    smiles = valid_df[[smiles_col, "solvent"]].values.tolist()

    for target in targets:
        model_path = REPO_ROOT / "model" / "flsf" / f"FluoDB_{target}" / "fold_0" / "model_0"
        df[f"{target}_pred"] = np.nan
        df.loc[valid_mask, f"{target}_pred"] = flsf_predict(
            model_path=str(model_path),
            smiles=smiles,
        )

    df.to_csv(output_file, index=False)
    print(f"Valid SMILES: {int(valid_mask.sum())}")
    print(f"Invalid SMILES skipped: {int((~valid_mask).sum())}")
    print(f"Saved combined predictions to {output_file}")
