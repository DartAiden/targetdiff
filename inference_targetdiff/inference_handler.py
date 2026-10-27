import json
import os
import re
import sys
import tarfile
from typing import Any, Tuple

import torch
from rdkit import Chem
from torch_geometric.transforms import Compose
from ts.handler_utils.timer import timed

from qbio.file_handler import FileHandler
from qbio.inference_handler import InferenceHandler
from qbio.types import OutputTypes

from inference_targetdiff.inference_types import InferenceInput, TargetDiffConfig


class TargetDiffHandler(InferenceHandler):
    def __init__(self):
        super().__init__()
        self.device = None
        self.out_dir = None
        self.model = None
        self.transform = None
        self.reconstruct = None
        self.trans = None

    def initialize(self, context):
        super().initialize(context)
        source_archive = os.path.join(self.model_dir, "targetdiff_source.tar.gz")
        source_root = os.path.join(self.model_dir, "targetdiff_source")
        os.makedirs(source_root, exist_ok=True)
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(source_root)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        import utils.reconstruct as reconstruct
        import utils.transforms as trans
        from models.molopt_score_model import ScorePosNet3D

        self.reconstruct = reconstruct
        self.trans = trans
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.join(self.model_dir, "targetdiff.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint["config"]
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(
            model_config.data.transform.ligand_atom_mode
        )
        self.transform = Compose([protein_featurizer])
        self.model = ScorePosNet3D(
            model_config.model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @timed
    def preprocess(self, data) -> Tuple[Any, TargetDiffConfig, InferenceInput]:
        from datasets.pl_data import ProteinLigandData, torchify_dict
        from utils.data import PDBProtein

        model_input = super().validate_input(data, InferenceInput)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model_input.id):
            raise ValueError("Request id contains unsupported filename characters.")

        input_files = FileHandler.get_inference_input_files(
            model_input,
            self.env_config.LOCAL_EXECUTION,
            [".pdb", ".json"],
        )
        pdb_files = input_files[".pdb"]
        config_files = input_files[".json"]
        if len(pdb_files) != 1:
            raise ValueError("Provide exactly one protein pocket PDB file.")
        if len(config_files) > 1:
            raise ValueError("Provide at most one JSON sampling configuration.")

        config = TargetDiffConfig()
        if config_files:
            try:
                with open(config_files[0], "r", encoding="utf-8") as config_file:
                    config = TargetDiffConfig.model_validate(json.load(config_file))
            except json.JSONDecodeError as error:
                raise ValueError("Sampling configuration must be valid JSON.") from error

        pdb_file = pdb_files[0]
        self.out_dir = os.path.join(
            os.path.dirname(os.path.dirname(str(pdb_file))),
            model_input.output_folder,
        )
        os.makedirs(self.out_dir, exist_ok=True)
        pocket_dict = PDBProtein(str(pdb_file)).to_dict_atom()
        graph = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict={
                "element": torch.empty([0], dtype=torch.long),
                "pos": torch.empty([0, 3], dtype=torch.float),
                "atom_feature": torch.empty([0, 8], dtype=torch.float),
                "bond_index": torch.empty([2, 0], dtype=torch.long),
                "bond_type": torch.empty([0], dtype=torch.long),
            },
        )
        graph = self.transform(graph)
        if graph.protein_pos.size(0) == 0:
            raise ValueError("PDB does not contain a valid heavy-atom pocket.")
        return graph, config, model_input

    @timed
    def inference(
        self, data: Tuple[Any, TargetDiffConfig, InferenceInput], *args, **kwargs
    ):
        from scripts.sample_diffusion import sample_diffusion_ligand
        from utils.misc import seed_all

        graph, config, model_input = data
        seed_all(config.seed)
        with torch.no_grad():
            positions, atom_types, *_ = sample_diffusion_ligand(
                self.model,
                graph,
                config.num_samples,
                batch_size=config.batch_size,
                device=str(self.device),
                num_steps=config.num_steps,
                pos_only=False,
                center_pos_mode="protein",
                sample_num_atoms="prior",
            )

        molecules = []
        reconstructed_count = 0
        for predicted_positions, predicted_types in zip(positions, atom_types):
            atomic_numbers = self.trans.get_atomic_number_from_index(
                predicted_types, mode="add_aromatic"
            )
            try:
                aromatic = self.trans.is_aromatic_from_index(
                    predicted_types, mode="add_aromatic"
                )
                molecule = self.reconstruct.reconstruct_from_generated(
                    predicted_positions, atomic_numbers, aromatic
                )
            except self.reconstruct.MolReconsError:
                molecules.append(None)
                continue
            reconstructed_count += 1
            if "." in Chem.MolToSmiles(molecule):
                molecules.append(None)
            else:
                molecules.append(molecule)
        return molecules, reconstructed_count, config, model_input

    @timed
    def postprocess(self, data):
        molecules, reconstructed_count, config, model_input = data
        request_id = model_input.id
        sdf_path = os.path.join(self.out_dir, f"targetdiff_{request_id}.sdf")
        smiles_path = os.path.join(self.out_dir, f"targetdiff_{request_id}.smi")
        summary_path = os.path.join(
            self.out_dir, f"targetdiff_summary_{request_id}.json"
        )
        smiles = []
        with open(sdf_path, "w", encoding="utf-8") as sdf_file, open(
            smiles_path, "w", encoding="utf-8"
        ) as smiles_file:
            for index, molecule in enumerate(molecules):
                if molecule is None:
                    continue
                canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
                molecule.SetProp("_Name", f"TargetDiff_{index}")
                sdf_file.write(Chem.MolToMolBlock(molecule) + "\n$$$$\n")
                smiles_file.write(canonical_smiles + "\n")
                smiles.append(canonical_smiles)

        unique_count = len(set(smiles))
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(
                {
                    "model": "targetdiff",
                    "requested_samples": config.num_samples,
                    "reconstructed_samples": reconstructed_count,
                    "valid_samples": len(smiles),
                    "unique_valid_samples": unique_count,
                    "validity_rate": len(smiles) / config.num_samples,
                    "uniqueness_rate": unique_count / len(smiles) if smiles else 0.0,
                    "sampling_parameters": config.model_dump(),
                    "interpretation": (
                        "Molecule reconstruction metrics only; they do not measure "
                        "binding affinity or biological activity."
                    ),
                },
                summary_file,
                indent=2,
            )

        archive_path = os.path.join(
            self.out_dir, f"targetdiff_results_{request_id}.zip"
        )
        FileHandler.archive_files(
            [sdf_path, smiles_path, summary_path], archive_path, self.out_dir
        )
        output_location = FileHandler.upload_files_to_bucket_folder(
            model_input, archive_path, self.env_config.LOCAL_EXECUTION
        )
        summary_location = FileHandler.upload_files_to_bucket_folder(
            model_input, summary_path, self.env_config.LOCAL_EXECUTION
        )
        return super().prepare_inference_response(
            model_input,
            "targetdiff",
            {
                OutputTypes.OUTPUT.value: output_location,
                OutputTypes.OUTPUT_SUMMARY.value: summary_location,
            },
        )
