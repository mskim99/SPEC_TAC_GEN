import os
import glob
import argparse
import numpy as np
import csv
import random
from typing import List, Dict


def parse_material(path: str) -> str:
    base = os.path.basename(path)
    if "_Movement_" not in base:
        raise ValueError(f"Unexpected filename (missing '_Movement_'): {base}")
    return base.split("_Movement_")[0]


def collect_files(input_dir: str, recursive: bool) -> List[str]:
    pattern = "**/*.npy" if recursive else "*.npy"
    files = sorted(glob.glob(os.path.join(input_dir, pattern), recursive=recursive))
    return files


def group_by_material(files: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for f in files:
        m = parse_material(f)
        groups.setdefault(m, []).append(f)
    # Deterministic order in each group
    for m in groups:
        groups[m] = sorted(groups[m])
    return groups


def choose_materials(all_materials: List[str],
                     specified: List[str],
                     max_materials: int,
                     select_mode: str,
                     seed: int) -> List[str]:
    materials_sorted = sorted(all_materials)

    if specified:
        chosen = []
        missing = []
        sset = set(specified)
        for m in materials_sorted:
            if m in sset:
                chosen.append(m)
        for m in specified:
            if m not in set(materials_sorted):
                missing.append(m)
        if missing:
            print(f"[WARN] These materials were not found and will be ignored: {missing}")
        return chosen

    if max_materials is None or max_materials <= 0 or max_materials >= len(materials_sorted):
        return materials_sorted

    if select_mode == "random":
        rng = random.Random(seed)
        chosen = materials_sorted[:]
        rng.shuffle(chosen)
        chosen = sorted(chosen[:max_materials])  # Sort for stable folder order
        return chosen

    # Default: first N (alphabetical)
    return materials_sorted[:max_materials]


def main():
    ap = argparse.ArgumentParser(description="Extract overlapping or non-overlapping chunks from .npy signals.")
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing the original .npy files")
    ap.add_argument("--output_dir", type=str, default="./data/spectrogram/LMTHaptic",
                    help="Output directory (default: ./LMTHaptic)")
    ap.add_argument("--chunk_len", type=int, default=256,
                    help="Length of each chunk (default: 256)")
    # Stride option: default 0 means it behaves identically to chunk_len (non-overlapping)
    ap.add_argument("--stride", type=int, default=0,
                    help="Sliding window stride (default: 0, which sets stride equal to chunk_len for non-overlapping chunks)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recursively search for .npy files in subdirectories")

    # Material selection options
    ap.add_argument("--max_materials", type=int, default=0,
                    help="Limit the number of materials to process (0 for all materials)")
    ap.add_argument("--select", choices=["first", "random"], default="first",
                    help="Selection mode when using max_materials (first|random)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for random selection")
    ap.add_argument("--materials", nargs="*", default=[],
                    help="Explicitly specify materials to save (e.g., --materials G1EpoxyRasterPlate G1FineAluminumMesh)")
    
    # Limit maximum chunks per source file
    ap.add_argument("--max_chunks_per_file", type=int, default=0,
                    help="Maximum number of chunks to save per source file (0 for no limit)")

    args = ap.parse_args()

    # If stride is 0 or not provided, set it to chunk_len
    stride = args.stride if args.stride > 0 else args.chunk_len

    files = collect_files(args.input_dir, args.recursive)
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    groups = group_by_material(files)
    all_materials = list(groups.keys())

    chosen_materials = choose_materials(
        all_materials=all_materials,
        specified=args.materials,
        max_materials=args.max_materials,
        select_mode=args.select,
        seed=args.seed
    )

    if not chosen_materials:
        raise ValueError("No materials selected. Check --materials names or --max_materials.")

    print(f"[INFO] Found materials: {len(all_materials)}")
    print(f"[INFO] Selected materials: {len(chosen_materials)} -> {chosen_materials}")
    print(f"[INFO] Chunk Length: {args.chunk_len}, Stride: {stride}")

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        # Recording start_idx is useful for future restoration or analysis.
        w.writerow(["material", "src_file", "chunk_file", "chunk_index_in_src", "chunk_len", "src_total_len", "start_idx"])

        total_saved = 0
        total_skipped_short = 0

        for material in chosen_materials:
            mfiles = groups.get(material, [])
            mat_out_dir = os.path.join(args.output_dir, material)
            os.makedirs(mat_out_dir, exist_ok=True)

            print(f"\n[Material] {material}  (files: {len(mfiles)})")

            for src in mfiles:
                arr = np.load(src, allow_pickle=False)

                if arr.ndim == 0:
                    print(f"  - skip (scalar?): {os.path.basename(src)}")
                    continue

                T = arr.shape[0]
                
                if T < args.chunk_len:
                    total_skipped_short += 1
                    print(f"  - skip (too short: {T}): {os.path.basename(src)}")
                    continue
                
                # Calculate possible chunks based on stride
                n_chunks_possible = ((T - args.chunk_len) // stride) + 1

                # Apply limit (0 means unlimited)
                n_chunks = n_chunks_possible
                if args.max_chunks_per_file and args.max_chunks_per_file > 0:
                    n_chunks = min(n_chunks_possible, args.max_chunks_per_file)

                # The end length of the mathematically extractable region
                usable = (n_chunks - 1) * stride + args.chunk_len if n_chunks > 0 else 0
                src_base = os.path.splitext(os.path.basename(src))[0]

                for i in range(n_chunks):
                    # Calculate the start position of the i-th chunk using stride
                    s = i * stride
                    e = s + args.chunk_len
                    chunk = arr[s:e]

                    chunk_name = f"{src_base}_chunk{i:06d}.npy"
                    out_path = os.path.join(mat_out_dir, chunk_name)
                    np.save(out_path, chunk)

                    w.writerow([material, src, os.path.join(material, chunk_name), i, args.chunk_len, T, s])
                    total_saved += 1

                print(
                    f"  - {os.path.basename(src)}: T={T} -> "
                    f"saved={n_chunks}/{n_chunks_possible} "
                    f"(usable={usable}, dropped={T-usable})"
                )

        print(f"\nDone.")
        print(f"Saved chunks: {total_saved}")
        print(f"Skipped short files (<{args.chunk_len}): {total_skipped_short}")
        print(f"Output dir: {os.path.abspath(args.output_dir)}")
        print(f"Manifest: {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()