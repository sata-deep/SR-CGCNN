#!/usr/bin/env python
"""Download a CGCNN formation-energy dataset from Materials Project.

The script writes the ordinary CGCNN dataset layout into this directory:

    id_prop.csv       rows: material_id, formation_energy_per_atom
    atom_init.json    copied from ../SR-CGCNN/cgcnn/atom_init.json
    mp-*.cif          one CIF file per material

Example:
    export MP_API_KEY="your-new-key"
    python download.py --limit 1000 --stable-only
"""

from __future__ import print_function

import argparse
import csv
import inspect
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ATOM_INIT = PROJECT_DIR / "SR-CGCNN" / "cgcnn" / "atom_init.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Materials Project formation energies and CIFs "
                    "as a CGCNN dataset.")
    parser.add_argument(
        "--api-key", default=os.environ.get("MP_API_KEY"),
        help="Materials Project API key. Defaults to MP_API_KEY.")
    parser.add_argument(
        "--output-dir", default=str(SCRIPT_DIR),
        help="Directory for id_prop.csv, atom_init.json, and CIF files "
             "(default: this FE directory).")
    parser.add_argument(
        "--atom-init-source", default=str(DEFAULT_ATOM_INIT),
        help="Path to atom_init.json to copy into the dataset directory.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of materials to save. Omit to download all "
             "matches returned by the query.")
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Materials Project query chunk size.")
    parser.add_argument(
        "--stable-only", action="store_true",
        help="Only download materials on the convex hull.")
    parser.add_argument(
        "--include-gnome", action="store_true",
        help="Include GNoME materials in the summary search.")
    parser.add_argument(
        "--formation-energy-min", type=float, default=None,
        help="Minimum formation energy per atom in eV/atom.")
    parser.add_argument(
        "--formation-energy-max", type=float, default=None,
        help="Maximum formation energy per atom in eV/atom.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rewrite existing CIF files.")
    return parser.parse_args()


def copy_atom_init(source, output_dir):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError("atom_init.json source not found: {}".format(
            source))
    shutil.copyfile(str(source), str(output_dir / "atom_init.json"))


def formation_energy_window(args):
    if args.formation_energy_min is None and args.formation_energy_max is None:
        return None
    return (args.formation_energy_min, args.formation_energy_max)


def query_materials(args):
    if not args.api_key:
        raise RuntimeError(
            "No API key found. Set MP_API_KEY or pass --api-key.")

    chunk_size = args.chunk_size
    if args.limit is not None:
        chunk_size = min(args.chunk_size, args.limit)

    num_chunks = None
    if args.limit is not None:
        num_chunks = int(math.ceil(float(args.limit) / chunk_size))

    criteria = {
        "fields": ["material_id", "structure", "formation_energy_per_atom"],
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
    }
    if args.stable_only:
        criteria["is_stable"] = True
    fe_window = formation_energy_window(args)
    if fe_window is not None:
        criteria["formation_energy"] = fe_window

    with MPRester(args.api_key) as mpr:
        search_params = inspect.signature(
            mpr.materials.summary.search).parameters
        if "include_gnome" in search_params:
            criteria["include_gnome"] = args.include_gnome
        elif args.include_gnome:
            raise RuntimeError(
                "This installed mp-api version does not support "
                "--include-gnome. Upgrade mp-api or omit that option.")
        docs = mpr.materials.summary.search(**criteria)

    if args.limit is not None:
        docs = docs[:args.limit]
    return docs


def get_doc_value(doc, name):
    if isinstance(doc, dict):
        return doc[name]
    return getattr(doc, name)


def write_dataset(docs, output_dir, overwrite=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    id_prop_path = output_dir / "id_prop.csv"

    count = 0
    with id_prop_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        for doc in docs:
            material_id = str(get_doc_value(doc, "material_id"))
            formation_energy = get_doc_value(doc, "formation_energy_per_atom")
            structure = get_doc_value(doc, "structure")
            cif_path = output_dir / "{}.cif".format(material_id)

            if overwrite or not cif_path.exists():
                cif_path.write_text(str(CifWriter(structure)), encoding="utf-8")

            writer.writerow([material_id, formation_energy])
            count += 1

    return count


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_atom_init(args.atom_init_source, output_dir)
    docs = query_materials(args)
    count = write_dataset(docs, output_dir, overwrite=args.overwrite)

    print("Saved {} materials to {}".format(count, output_dir))
    print("Target property: formation_energy_per_atom (eV/atom)")
    print("Files: id_prop.csv, atom_init.json, and mp-*.cif")


if __name__ == "__main__":
    main()
