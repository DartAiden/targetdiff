import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument("--proteinpath", required=True)
parser.add_argument("--ligandpath", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--radius", type=float, default=10.0)
parser.add_argument("--outdir", default="/targetdiff/prepared")
parser.add_argument("--repo", default="/targetdiff")
args = parser.parse_args()

root = Path(args.repo).resolve()
sys.path.insert(0, str(root))

import yaml
from utils.data import PDBProtein
from datasets.protein_ligand import parse_sdf_file_mol

outdir = Path(args.outdir).resolve()
outdir.mkdir(parents=True, exist_ok=True)

protein = PDBProtein(str(Path(args.proteinpath).resolve()))
ligand = parse_sdf_file_mol(str(Path(args.ligandpath).resolve()))

residues = protein.query_residues_ligand(ligand, args.radius)
if not residues:
    raise ValueError("No pocket residues found. Check protein–ligand alignment.")

pocket_path = outdir / "pocket.pdb"
pocket_path.write_text(protein.residues_to_pdb_block(residues))

config = yaml.safe_load((root / "configs/sampling.yml").read_text())
config["model"]["checkpoint"] = str(Path(args.checkpoint).resolve())

config_path = outdir / "sampling.yml"
config_path.write_text(yaml.safe_dump(config))

print(f"Pocket: {pocket_path}")
print(f"Config: {config_path}")