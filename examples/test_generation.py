from pathlib import Path
import yaml
from utils.data import PDBProtein
from datasets.protein_ligand import parse_sdf_file

root = Path("/content/targetdiff")
protein = PDBProtein(str(root / "examples/3ug2_protein.pdb"))
ligand = parse_sdf_file(str(root / "examples/3ug2_ligand.sdf"))

residues = protein.query_residues_ligand(ligand, 10)
pocket = root / "examples/3ug2_pocket10.pdb"
pocket.write_text(protein.residues_to_pdb_block(residues))

config = yaml.safe_load((root / "configs/sampling.yml").read_text())
config["model"]["checkpoint"] = str(root / "pretrained_models/targetdiff.pt")
(root / "configs/sampling_3ug2.yml").write_text(yaml.safe_dump(config))

print("Pocket and sampling configuration saved.")