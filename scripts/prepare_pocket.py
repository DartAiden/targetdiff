import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--proteinpath", required=True)
parser.add_argument("--ligandpath", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--radius", type=float, default=10.0)
parser.add_argument("--outdir", default=None)
parser.add_argument("--repo", default=None)
args = parser.parse_args()

root = (
    Path(args.repo).expanduser().resolve()
    if args.repo
    else Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(root))
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import yaml
from utils.data import PDBProtein
from datasets.protein_ligand import parse_sdf_file_mol

protein_path = Path(args.proteinpath).expanduser().resolve()
ligand_path = Path(args.ligandpath).expanduser().resolve()
checkpoint = Path(args.checkpoint).expanduser().resolve()

for path in (protein_path, ligand_path, checkpoint):
    if not path.is_file():
        raise FileNotFoundError(path)

outdir = (
    Path(args.outdir).expanduser().resolve()
    if args.outdir
    else root / "prepared"
)
outdir.mkdir(parents=True, exist_ok=True)

protein = PDBProtein(str(protein_path))
ligand = parse_sdf_file_mol(str(ligand_path))

residues = protein.query_residues_ligand(ligand, args.radius)
if not residues:
    raise ValueError("No pocket residues found. Check protein–ligand alignment.")

pocket_path = outdir / "pocket.pdb"
pocket_path.write_text(protein.residues_to_pdb_block(residues))

config = yaml.safe_load((root / "configs/sampling.yml").read_text())
config["model"]["checkpoint"] = str(checkpoint)

config_path = outdir / "sampling.yml"
config_path.write_text(yaml.safe_dump(config, sort_keys=False))

print(f"Pocket: {pocket_path}")
print(f"Config: {config_path}")