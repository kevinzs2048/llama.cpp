"""Report code-object resources without loading a GPU module or importing HIP."""

import argparse
import json
from pathlib import Path

from flagos_amd_hsaco import read_hsaco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hsaco", type=Path, nargs="+")
    args = parser.parse_args()
    records = []
    for path in args.hsaco:
        metadata = read_hsaco(path)
        records.append({
            "file": str(path),
            "target": metadata.target,
            "symbol": metadata.symbol,
            "wavefront_size": metadata.wavefront_size,
            "max_flat_workgroup_size": metadata.max_flat_workgroup_size,
            "fixed_group_segment_bytes": metadata.group_segment_fixed_size,
            "private_segment_bytes_per_workitem": metadata.private_segment_fixed_size,
            "vgpr_count": metadata.vgpr_count,
            "sgpr_count": metadata.sgpr_count,
            "vgpr_spill_count": metadata.vgpr_spill_count,
            "sgpr_spill_count": metadata.sgpr_spill_count,
        })
    print(json.dumps({
        "scope": "ELF resources only; no occupancy or performance measurement",
        "kernels": records,
    }, indent=2))


if __name__ == "__main__":
    main()
