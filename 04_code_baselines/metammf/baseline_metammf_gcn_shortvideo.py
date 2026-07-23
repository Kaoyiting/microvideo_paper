#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MetaMMF-GCN baseline for ShortVideo."""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(ARTIFACT_ROOT, ".."))
SHARED_SRC = os.path.join(ARTIFACT_ROOT, "04_code_baselines", "shared_src")
for path in [SHARED_SRC, os.path.join(REPO_ROOT, "Interaction", "src"), os.path.join(REPO_ROOT, "content")]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dataloader import ShortVideoDataset
from metammf_baseline_common import (
    build_eval_data_shortvideo,
    load_shortvideo_aligned,
    load_three_modal_features,
    run_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MetaMMF-GCN baseline for ShortVideo.")
    parser.add_argument("--data-dir", default=os.path.join(ARTIFACT_ROOT, "01_data_processed", "shortvideo"))
    parser.add_argument(
        "--modal-embeddings",
        default=os.path.join(
            ARTIFACT_ROOT,
            "02_features_embeddings",
            "content_imagebind_embeddings",
            "shortvideo_dualgnn_three_modal_embeddings.npz",
        ),
    )
    parser.add_argument(
        "--aligned-embeddings",
        default=os.path.join(
            ARTIFACT_ROOT,
            "02_features_embeddings",
            "aligned_threepath_embeddings",
            "shortvideo_threepath_aligned_embeddings_1024dim.npz",
        ),
    )
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "best_metammf_gcn_shortvideo.pth"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="Debug only: cap training batches per epoch.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--signal-dim", type=int, default=5)
    parser.add_argument("--fusion-dims", type=int, nargs="+", default=[1024, 512, 256, 32])
    parser.add_argument("--item-id-dim", type=int, default=32)
    parser.add_argument("--sage-dim", type=int, default=64)
    parser.add_argument("--dropout1", type=float, default=0.5)
    parser.add_argument("--dropout2", type=float, default=0.2)
    parser.add_argument("--fusion-batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def solve():
    args = parse_args()
    print("=" * 80)
    print("MetaMMF-GCN Baseline for ShortVideo")
    print("Official dynamic visual/acoustic/textual fusion + two-layer GraphSAGE")
    print("=" * 80)

    dataset = ShortVideoDataset(path=args.data_dir)
    features = load_three_modal_features(args.modal_embeddings, dataset.m_items)
    aligned_items, item_id_to_aligned_idx, _, missing_count, original_count = load_shortvideo_aligned(
        args.aligned_embeddings,
        dataset.m_items,
    )
    eval_data = build_eval_data_shortvideo(dataset, item_id_to_aligned_idx)
    print(
        f"Users={dataset.n_users}, Items={dataset.m_items}, "
        f"aligned_items={len(aligned_items)}/{original_count}, missing={missing_count}"
    )
    print(
        "Feature shapes: "
        f"visual={features['visual'].shape}, acoustic={features['acoustic'].shape}, textual={features['textual'].shape}"
    )

    run_training(
        args,
        dataset,
        features,
        np.asarray(aligned_items, dtype=np.int64),
        eval_data,
        {
            "model_name": "MetaMMF-GCN (ShortVideo)",
            "dataset": "ShortVideo",
            "candidate_space": "valid_ids from shortvideo_threepath_aligned_embeddings_1024dim.npz",
            "adaptation": "Official three-modal MetaMMF-GCN path on local visual/acoustic/textual item embeddings.",
        },
    )


if __name__ == "__main__":
    solve()
