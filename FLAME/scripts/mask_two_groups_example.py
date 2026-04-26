import argparse
import os

from FLAME import flsf_group_mask_explain_all_targets


FLUOROPHORE_SMILES = "CCN(CC)c6ccc5[n-]c4c(cc([O-])c3ccc(Oc2c([O-])c1ccccc1c([O-])c2Cl)cc34)oc5c6"
SOLVENT_SMILES = "CS(C)=O"

# Group I is the left chlorinated fragment in the user's figure.
GROUP_I_ATOMS = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Group II is the right diethylamino fragment in the user's figure.
GROUP_II_ATOMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 33, 34, 35, 36, 37]

MASK_GROUPS = {
    "group_I_red": GROUP_I_ATOMS,
    "group_II_blue": GROUP_II_ATOMS,
}


def main():
    parser = argparse.ArgumentParser(
        description='Mask the two predefined functional groups of the specified fluorophore-solvent pair and compare FLSF predictions.'
    )
    parser.add_argument('--model_root', default='model/flsf', help='Root directory containing per-target FLSF model folders.')
    parser.add_argument('--model_prefix', default='FluoDB', help='Model prefix used in per-target folder names, e.g. FluoDB_emi.')
    parser.add_argument('--targets', nargs='+', default=['abs', 'emi', 'plqy', 'e'], help='Targets to evaluate.')
    parser.add_argument('--output_dir', default='pred/flsf_group_mask', help='Directory for CSV and group-highlight PNG outputs.')
    parser.add_argument('--prefix', default='two_group_mask', help='Output CSV filename prefix.')
    parser.add_argument('--no_cuda', action='store_true', help='Force CPU inference.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'{args.prefix}.csv')
    image_dir = os.path.join(args.output_dir, 'group_highlights')

    result = flsf_group_mask_explain_all_targets(
        model_root=args.model_root,
        fluorophore_smiles=FLUOROPHORE_SMILES,
        solvent_smiles=SOLVENT_SMILES,
        mask_groups=MASK_GROUPS,
        model_prefix=args.model_prefix,
        targets=args.targets,
        output_csv=csv_path,
        output_png_dir=image_dir,
        no_cuda=args.no_cuda
    )

    print(result.to_string(index=False))
    print(f'\nSaved CSV to {csv_path}')
    print(f'Saved group highlight images to {image_dir}')


if __name__ == '__main__':
    main()
