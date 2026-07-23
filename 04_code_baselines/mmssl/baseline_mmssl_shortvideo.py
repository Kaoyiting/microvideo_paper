#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMSSL baseline for ShortVideo.

Reference:
    Wei et al. (2023), "Multi-Modal Self-Supervised Learning for
    Recommendation", WWW 2023.

Official implementation:
    https://github.com/HKUDS/MMSSL

Implementation notes:
    - This script follows the official MMSSL image/text training path and maps
      local ShortVideo visual features to image and textual features to text.
    - Acoustic features are not used because the official MMSSL implementation
      in HKUDS/MMSSL defines image/text encoders only.
    - The official code depends on CUDA+DGL. This repository version is
      device-aware and DGL-free while preserving the official main loss path.
"""

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
from mmssl_baseline_common import (
    build_eval_data_shortvideo,
    load_shortvideo_aligned,
    load_visual_textual_features,
    run_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MMSSL baseline for ShortVideo.")
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
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "best_mmssl_shortvideo.pth"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0, help="Debug only: cap training batches per epoch.")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--lr", type=float, default=0.00055)
    parser.add_argument("--d-lr", type=float, default=3e-4)
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--weight-size", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--model-cat-rate", type=float, default=0.55)
    parser.add_argument("--id-cat-rate", type=float, default=0.36)
    parser.add_argument("--head-num", type=int, default=4)
    parser.add_argument("--feat-reg-decay", type=float, default=1e-5)
    parser.add_argument("--cl-rate", type=float, default=0.03)
    parser.add_argument("--g-rate", type=float, default=0.0001)
    parser.add_argument("--g-drop1", type=float, default=0.31)
    parser.add_argument("--g-drop2", type=float, default=0.5)
    parser.add_argument("--gp-rate", type=float, default=1.0)
    parser.add_argument("--real-data-tau", type=float, default=0.005)
    parser.add_argument("--log-log-scale", type=float, default=0.00001)
    parser.add_argument("--ui-pre-scale", type=float, default=100.0)
    parser.add_argument("--m-topk-rate", type=float, default=0.0001)
    parser.add_argument("--t-update", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--cl-batch-size", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def solve():
    args = parse_args()
    print("=" * 80)
    print("MMSSL Baseline for ShortVideo")
    print("Official image/text path mapped to local visual/textual features")
    print("=" * 80)

    dataset = ShortVideoDataset(path=args.data_dir)
    image_feats, text_feats = load_visual_textual_features(args.modal_embeddings, dataset.m_items)
    aligned_items, item_id_to_aligned_idx, _, missing_count, original_count = load_shortvideo_aligned(
        args.aligned_embeddings,
        dataset.m_items,
    )
    eval_data = build_eval_data_shortvideo(dataset, item_id_to_aligned_idx)
    print(
        f"Users={dataset.n_users}, Items={dataset.m_items}, "
        f"aligned_items={len(aligned_items)}/{original_count}, missing={missing_count}"
    )
    print(f"Feature shapes: image={image_feats.shape}, text={text_feats.shape}")

    run_training(
        args,
        dataset,
        image_feats,
        text_feats,
        np.asarray(aligned_items, dtype=np.int64),
        eval_data,
        {
            "model_name": "MMSSL (ShortVideo)",
            "dataset": "ShortVideo",
            "adaptation": "Official image/text MMSSL path; local visual/textual features; acoustic omitted for source fidelity.",
            "candidate_space": "valid_ids from shortvideo_threepath_aligned_embeddings_1024dim.npz",
        },
    )


if __name__ == "__main__":
    solve()
