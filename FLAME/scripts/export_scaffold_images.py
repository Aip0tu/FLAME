import argparse
import importlib.util
import os
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def load_scaffold_dict(scaffold_file: Path):
    spec = importlib.util.spec_from_file_location('flsf_scaffold_module', scaffold_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.scaffold


def safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')


def draw_scaffold_png(smiles: str, output_path: Path, legend: str = '', size: int = 800):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    options = drawer.drawOptions()
    options.legendFontSize = 24
    options.addStereoAnnotation = False
    options.bondLineWidth = 2
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(drawer.GetDrawingText())
    return True


def export_scaffold_images(output_dir: Path,
                           scaffold_file: Path,
                           image_size: int = 800):
    scaffold = load_scaffold_dict(scaffold_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = []
    for class_name, smiles_list in scaffold.items():
        class_dir = output_dir / safe_name(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        for index, smiles in enumerate(smiles_list, start=1):
            file_name = f'{index:03d}_{safe_name(class_name)}.png'
            legend = f'{class_name} #{index}'
            ok = draw_scaffold_png(
                smiles=smiles,
                output_path=class_dir / file_name,
                legend=legend,
                size=image_size
            )
            if ok:
                total += 1
            else:
                skipped.append({
                    'class_name': class_name,
                    'index': index,
                    'smiles': smiles
                })

    return len(scaffold), total, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Export all fluorophore scaffolds from scaffold.py into one top-level folder with 16 category subfolders.'
    )
    parser.add_argument(
        '--output_dir',
        default='pred/scaffold_gallery',
        help='Top-level output directory. Sixteen class folders will be created inside it.'
    )
    parser.add_argument(
        '--scaffold_file',
        default='FLAME/dataprocess/flsf/scaffold.py',
        help='Path to scaffold.py.'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=800,
        help='PNG width and height in pixels.'
    )
    args = parser.parse_args()

    scaffold_file = Path(args.scaffold_file).resolve()
    output_dir = Path(args.output_dir).resolve()

    num_classes, total, skipped = export_scaffold_images(
        output_dir=output_dir,
        scaffold_file=scaffold_file,
        image_size=args.image_size
    )

    print(f'Exported {total} scaffold images into {num_classes} folders under {output_dir}')
    if skipped:
        print(f'Skipped {len(skipped)} invalid scaffold SMILES:')
        for item in skipped:
            print(f'  {item["class_name"]} #{item["index"]}: {item["smiles"]}')


if __name__ == '__main__':
    main()
