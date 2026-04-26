import argparse
import os
import re

import pandas as pd

from FLAME import flsf_atom_explain, flsf_atom_explain_all_targets


# EXAMPLE_FLUOROPHORES = [
#     "Cc2coc3c(Cl)cc1c(C)cc(=O)oc1c23",
#     "CN(C)c5ccc4nc(/C=C/c1cc2CCCN3CCCc(c1)c23)c(=O)oc4c5",
#     "O=c3oc2ccc1ccc(O)cc1c2nc3c4ccccc4",
#     "N#Cc2cc1ccccc1oc2=O",
#     "CCN(CC)c2ccc1cc(C#N)c(=NC(=O)C(C)C)oc1c2",
#     "N#Cc2cc1ccc(O)cc1oc2=N",
#     "CCN(CC)c3ccc2cc(C#N)c(=NC(=O)c1ccccc1)oc2c3"
# ]
#
# EXAMPLE_SOLVENTS = [
#     "O1CCOCC1",
#     "CCCCCCC",
#     "c1ccccc1",
#     "CCOC(=O)C",
#     "CCO",
#     "CS(C)=O",
#     "CCO"
# ]
EXAMPLE_FLUOROPHORES = [
    "Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1",
    "Cc1ccc(C(=O)c2cc(C(=O)O)cc3c2CCN3c2c(Cl)cccc2Cl)cc1",

]

EXAMPLE_SOLVENTS = [
    "ClCCl",
    "O",

]


def safe_file_stem(text: str, max_len: int = 80) -> str:
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('_')
    if not stem:
        stem = 'molecule'
    return stem[:max_len]


def main():
    parser = argparse.ArgumentParser(description='Run atom-level masking explainability for FLSF.')
    parser.add_argument('--model_path', help='Directory containing a single-target FLSF checkpoint or ensemble.')
    parser.add_argument('--model_root', default='model/flsf', help='Root directory containing per-target FLSF model folders.')
    parser.add_argument('--model_prefix', default='FluoDB', help='Model prefix used in per-target folder names, e.g. FluoDB_emi.')
    parser.add_argument('--targets', nargs='+', default=['abs', 'emi', 'plqy', 'e'], help='Targets to explain together.')
    parser.add_argument('--fluorophore', help='Single fluorophore SMILES.')
    parser.add_argument('--fluorophores', nargs='+', help='One or more fluorophore SMILES.')
    parser.add_argument('--solvents', nargs='+', help='One solvent SMILES per fluorophore.')
    parser.add_argument('--use_examples', action='store_true', help='Run the built-in seven fluorophore examples.')
    parser.add_argument('--solvent', default='O', help='Solvent SMILES. Defaults to water.')
    parser.add_argument('--output_dir', default='pred/flsf_explain', help='Directory for CSV and PNG outputs.')
    parser.add_argument('--prefix', default='flsf_atom_explain', help='Output file prefix.')
    parser.add_argument('--no_images', action='store_true', help='Skip PNG generation.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference.')
    args = parser.parse_args()

    fluorophores = []
    if args.fluorophore:
        fluorophores.append(args.fluorophore)
    if args.fluorophores:
        fluorophores.extend(args.fluorophores)
    entries = []

    if fluorophores:
        if args.solvents is not None:
            if len(args.solvents) != len(fluorophores):
                raise ValueError('The number of solvents must match the number of fluorophores.')
            solvents = list(args.solvents)
        else:
            solvents = [args.solvent] * len(fluorophores)
        entries.extend(zip(fluorophores, solvents))

    if args.use_examples or not entries:
        if len(EXAMPLE_FLUOROPHORES) != len(EXAMPLE_SOLVENTS):
            raise ValueError('EXAMPLE_FLUOROPHORES and EXAMPLE_SOLVENTS must have the same length.')
        entries.extend(zip(EXAMPLE_FLUOROPHORES, EXAMPLE_SOLVENTS))

    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.targets) == 1 and args.model_path:
        tables = []
        image_paths = []
        for idx, (fluorophore, solvent) in enumerate(entries, start=1):
            entry_prefix = f'{args.prefix}_{idx:02d}'
            csv_path = os.path.join(args.output_dir, f'{entry_prefix}.csv')
            png_path = os.path.join(
                args.output_dir,
                f'{entry_prefix}_{safe_file_stem(solvent)}.png'
            )
            result = flsf_atom_explain(
                model_path=args.model_path,
                fluorophore_smiles=fluorophore,
                solvent_smiles=solvent,
                output_csv=csv_path,
                output_png='' if args.no_images else png_path,
                no_cuda=args.no_cuda
            )
            result = result.copy()
            result.insert(0, 'entry_id', idx)
            tables.append(result)
            if not args.no_images:
                image_paths.append(png_path)

        merged = pd.concat(tables, ignore_index=True)
        merged_csv_path = os.path.join(args.output_dir, f'{args.prefix}.csv')
        merged.to_csv(merged_csv_path, index=False)
        print(merged.sort_values(['entry_id', 'abs_contribution'], ascending=[True, False]).head(20).to_string(index=False))
        print(f'\nSaved merged CSV to {merged_csv_path}')
        if not args.no_images:
            print('Saved figures:')
            for image_path in image_paths:
                print(image_path)
        return

    tables = []
    image_dirs = []
    for idx, (fluorophore, solvent) in enumerate(entries, start=1):
        example_dir = os.path.join(
            args.output_dir,
            f'{idx:02d}_{safe_file_stem(fluorophore)}__{safe_file_stem(solvent)}'
        )
        merged = flsf_atom_explain_all_targets(
            model_root=args.model_root,
            fluorophore_smiles=fluorophore,
            solvent_smiles=solvent,
            model_prefix=args.model_prefix,
            targets=args.targets,
            output_png_dir='' if args.no_images else example_dir,
            no_cuda=args.no_cuda
        )
        merged = merged.copy()
        merged.insert(0, 'entry_id', idx)
        tables.append(merged)
        if not args.no_images:
            image_dirs.append(example_dir)

    result = pd.concat(tables, ignore_index=True)
    csv_path = os.path.join(args.output_dir, f'{args.prefix}.csv')
    result.to_csv(csv_path, index=False)

    rank_columns = [f'importance_rank_{target}' for target in args.targets]
    preview_columns = ['entry_id', 'fluorophore_smiles', 'solvent_smiles', 'atom_index', 'atom_symbol', 'overall_importance_rank'] + rank_columns
    print(result.loc[:, preview_columns].sort_values(['entry_id', 'overall_importance_rank']).head(30).to_string(index=False))
    print(f'\nSaved CSV to {csv_path}')
    if not args.no_images:
        print('Saved example images under:')
        for image_dir in image_dirs:
            print(image_dir)


if __name__ == '__main__':
    main()
