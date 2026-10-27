# TargetDiff

TargetDiff is a 3D equivariant diffusion model for target-aware molecule generation. This service follows the upstream single-pocket sampling workflow and uses the included pretrained checkpoint.

## Input

Provide one prepared protein binding-pocket PDB file. The upstream README describes pockets as approximately 10 angstrom regions around a reference ligand. An optional JSON file may override the sampling parameters:

```json
{
  "seed": 2021,
  "num_samples": 10,
  "num_steps": 100,
  "batch_size": 16
}
```

All fields are optional. `num_samples` is limited to 1-100, `num_steps` to 1-1000, and `batch_size` to 1-32.

## Output

The result ZIP contains an SDF file, a SMILES file, and a JSON summary. The summary reports reconstruction and uniqueness metrics only. It does not measure binding affinity, biological activity, or experimental success.

## Upstream

- [TargetDiff repository](https://github.com/DartAiden/targetdiff)
- [TargetDiff paper](https://openreview.net/pdf?id=kJqXEPXMsE0)
