"""UTR guided-generation inference script — thin wrapper around guided_sampling.py.

Thesis alignment
================
Implements the inference pipeline for thesis tasks T5, T7a, T7b, T7c:

  T5  — predictor-guided DDPM sampling:
        ε̃(x_t, t) = ε_θ(x_t, t) + w · √(1−ᾱ_t) · ∇_{x_t} J(x_t; c)

  T7a — hard inpainting of the AUG start codon at the 3' end of the UTR.
        The last 3 positions (A=0, T/U=3, G=2) are frozen via RePaint-style
        re-injection at each reverse step.

  T7b — multi-objective J = α·f_TE(x_t) + β·J_gc(x_t) using the differentiable
        GC-content proxy (cosine similarity to G/C embeddings).

  T7c — guidance-scale sweep: run generation with several values of w in one
        job and save a separate CSV per scale for ablation figures.

Design principles
=================
* **Zero hardcoded values**: every model dim, vocab size, sequence length,
  guidance knob, and path comes from ``guided_config.yaml``.  The script reads
  the generator's architecture from the checkpoint's ``init_config`` key, so
  the YAML does not duplicate model dims.
* **Thin wrapper**: all reverse-diffusion math lives in ``guided_sampling.py``;
  this script only handles I/O (config, checkpoints, CSV, W&B).
* **Safe checkpoint loading**: uses ``UTRNoisyPredictor.from_checkpoint()`` and
  prefers the EMA embedding from the generator checkpoint.

Usage
-----
    cd ~/evodiff/Predictor/utr/guided_inference_utr
    python generate_optimized_utrs.py                        # default config
    python generate_optimized_utrs.py --config my_config.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup: make the Predictor root importable from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PREDICTOR_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_UTR_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

for _p in [_PREDICTOR_ROOT, _UTR_ROOT,
           os.path.join(_UTR_ROOT, "generator"),
           os.path.join(_UTR_ROOT, "noisy_predictor")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utr_model import UTRDiffusionUNet  # noqa: E402  (after path setup)
from utr_noisy_predictor_model import UTRNoisyPredictor  # noqa: E402
from utr_metric_utils import (  # noqa: E402
    decode_tokens, get_gc_content, get_max_homopolymer, get_mfe,
)
from guided_sampling import (  # noqa: E402
    build_noise_schedule,
    make_high_score_objective,
    make_regression_objective,
    make_hybrid_objective,
    make_multi_objective,
    guided_reverse_sample,
    score_generated_batch,
)


# ---------------------------------------------------------------------------
# Diversity metric (T6c / T7c)
# ---------------------------------------------------------------------------

def pairwise_hamming_diversity(sequences: List[str], max_seqs: int = 500) -> float:
    """Mean normalised pairwise Hamming diversity over generated sequences.

    Thesis alignment: T6c / T7c — measures how diverse the generated set is.
    Used in the guidance-scale sweep to show the diversity–quality trade-off
    (higher w → higher score but lower diversity).

    Returns a value in [0, 1]:
        0.0 = all sequences identical
        1.0 = every pair differs at every position

    Subsamples to ``max_seqs`` sequences when N is large.

    Args:
        sequences : list of equal-length nucleotide strings.
        max_seqs  : cap on the number of sequences used (random subsample).

    Returns:
        diversity : float in [0, 1].
    """
    n = len(sequences)
    if n < 2 or not sequences[0]:
        return 0.0

    if n > max_seqs:
        idx = np.random.choice(n, size=max_seqs, replace=False)
        sequences = [sequences[i] for i in idx]
        n = max_seqs

    L = len(sequences[0])
    arr = np.array([[ord(c) for c in s[:L]] for s in sequences], dtype=np.uint8)

    total_diff = 0
    count = 0
    for i in range(n):
        diffs = (arr[i + 1 :] != arr[i]).sum(axis=1)
        total_diff += int(diffs.sum())
        count += n - i - 1

    return float(total_diff) / (count * L) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Generator + embedding loader
# ---------------------------------------------------------------------------

def load_generator_and_embedding(
    checkpoint_path: str,
    device: torch.device,
):
    """Load UTRDiffusionUNet and token embedding from a generator checkpoint.

    Architecture is reconstructed from the checkpoint's ``init_config`` key so
    no model hyperparameters need to be duplicated in the guided_config.yaml.

    Returns:
        generator       : UTRDiffusionUNet in eval mode with requires_grad=False
        token_embedding : nn.Embedding (EMA weights preferred) in eval mode
        init_config     : dict — the generator's init_config (useful for embed_dim etc.)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # ---- reconstruct generator architecture from checkpoint -----------------
    # Try both key names: current trainers save "model_init_config"; older
    # checkpoints may have "init_config" or neither (pre-dates the feature).
    init_config = (
        ckpt.get("init_config")
        or ckpt.get("model_init_config")
    )
    if init_config is None:
        # Checkpoint predates init_config storage — fall back to the known
        # UTR generator architecture from utr_config.yaml.
        print(
            "[load_generator] WARNING: checkpoint has no init_config key. "
            "Falling back to default UTR architecture (embed_dim=256, "
            "encoder_channels=[256,512], max_len=1024)."
        )
        init_config = {
            "embed_dim": 256,
            "time_emb_dim": 1024,
            "max_len": 1024,
            "encoder_channels": [256, 512],
            "kernel_size": 5,
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
            "final_kernel_size": 3,
            "norm_groups": 8,
            "pool_factor": 2,
            "upsample_mode": "nearest",
            "time_emb_freq_base": 10000.0,
        }
    generator = UTRDiffusionUNet(**init_config).to(device)

    # Load EMA weights first (better than live weights for inference)
    model_state = ckpt.get("model_ema_state_dict") or ckpt.get("model_state_dict")
    if model_state is None:
        raise KeyError(
            f"Generator checkpoint '{checkpoint_path}' has neither "
            "'model_ema_state_dict' nor 'model_state_dict'."
        )
    generator.load_state_dict(model_state)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    # ---- load token embedding (prefer EMA) -----------------------------------
    vocab_size = ckpt.get("vocab_size") or ckpt.get("config", {}).get("data", {}).get("vocab_size", 7)
    embed_dim  = init_config["embed_dim"]

    token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
    emb_state = (
        ckpt.get("ema_embedding_state_dict")
        or ckpt.get("embedding_state_dict")
    )
    if emb_state is None:
        raise KeyError(
            f"Generator checkpoint '{checkpoint_path}' has neither "
            "'ema_embedding_state_dict' nor 'embedding_state_dict'."
        )
    token_embedding.load_state_dict(emb_state)
    token_embedding.to(device)
    token_embedding.eval()
    for p in token_embedding.parameters():
        p.requires_grad_(False)

    return generator, token_embedding, init_config


# ---------------------------------------------------------------------------
# Objective builder
# ---------------------------------------------------------------------------

def build_objective(
    cfg_gen: dict,
    predictor: UTRNoisyPredictor,
    token_embedding: torch.nn.Embedding,
    score_mean: float,
    score_std: float,
    device: torch.device,
):
    """Build the guidance objective function from generation config.

    Supports:
        "high_score" : BCE toward is_high=1 (default, T5a)
        "regression" : MSE toward target_mrl / score_std (T5b)
        "hybrid"     : weighted sum of both (T5c)
        "multi"      : TE + GC multi-objective (T7b)
                       automatically enabled when gc_weight != 0.
    """
    # T7b overrides the objective if gc_weight != 0
    gc_weight = float(cfg_gen.get("gc_weight", 0.0))
    if gc_weight != 0.0:
        te_weight = float(cfg_gen.get("te_weight", 1.0))
        target_gc = cfg_gen.get("target_gc")
        if target_gc is not None:
            target_gc = float(target_gc)
        # UTR vocab: A=0, C=1, G=2, T=3 → gc_token_ids = (G_id, C_id) = (2, 1)
        gc_token_ids = (2, 1)
        print(
            f"[objective] T7b multi-objective | te_weight={te_weight} "
            f"gc_weight={gc_weight} target_gc={target_gc}"
        )
        return make_multi_objective(
            te_weight=te_weight,
            gc_weight=gc_weight,
            token_embedding=token_embedding,
            gc_token_ids=gc_token_ids,
            target_gc=target_gc,
        )

    objective_name = cfg_gen.get("objective", "high_score")
    if objective_name == "high_score":
        print("[objective] T5 high-score BCE")
        return make_high_score_objective()
    elif objective_name == "regression":
        target_mrl  = float(cfg_gen.get("target_mrl", 8.5))
        target_norm = torch.tensor(
            (target_mrl - score_mean) / max(score_std, 1e-6),
            device=device,
        )
        print(f"[objective] T5 regression | target_mrl={target_mrl:.3f}")
        return make_regression_objective(target_norm)
    elif objective_name == "hybrid":
        target_mrl  = float(cfg_gen.get("target_mrl", 8.5))
        target_norm = torch.tensor(
            (target_mrl - score_mean) / max(score_std, 1e-6),
            device=device,
        )
        cls_w = float(cfg_gen.get("guidance_classification_weight", 1.0))
        reg_w = float(cfg_gen.get("guidance_regression_weight", 0.25))
        print(f"[objective] T5 hybrid | cls={cls_w} reg={reg_w} target_mrl={target_mrl:.3f}")
        return make_hybrid_objective(
            cls_weight=cls_w,
            reg_weight=reg_w,
            target_score_norm=target_norm,
        )
    else:
        raise ValueError(
            f"Unknown guidance objective '{objective_name}'. "
            "Choose from 'high_score', 'regression', 'hybrid'."
        )


# ---------------------------------------------------------------------------
# Frozen AUG mask builder (T7a)
# ---------------------------------------------------------------------------

def build_frozen_aug_masks(
    batch_size: int,
    seq_len: int,
    token_embedding: torch.nn.Embedding,
    device: torch.device,
):
    """Build frozen_mask and frozen_embeddings for the AUG start codon.

    In a 5'UTR of length ``seq_len``, the start codon AUG occupies the last
    3 positions: [seq_len-3, seq_len-2, seq_len-1].

    UTR vocab: A=0, C=1, G=2, T/U=3 → AUG = tokens [0, 3, 2]

    Returns:
        frozen_mask       : (B, L) bool tensor. True at the 3 AUG positions.
        frozen_embeddings : (B, D, L) float tensor. Zero everywhere except at
                            the AUG positions, where it holds the clean
                            token embedding.
    """
    # AUG = A(0), U→T(3), G(2) in the UTR vocab
    aug_token_ids = torch.tensor([0, 3, 2], dtype=torch.long, device=device)
    aug_embeddings = token_embedding(aug_token_ids)   # (3, D)

    frozen_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    frozen_mask[:, -3:] = True                         # last 3 positions

    # Build (B, D, L) frozen embedding tensor
    D = aug_embeddings.shape[-1]
    frozen_embeddings = torch.zeros(batch_size, D, seq_len, device=device)
    frozen_embeddings[:, :, -3:] = aug_embeddings.T.unsqueeze(0).expand(batch_size, -1, -1)

    return frozen_mask, frozen_embeddings


# ---------------------------------------------------------------------------
# Generation loop for one guidance scale
# ---------------------------------------------------------------------------

def generate_batch(
    scale: float,
    cfg: dict,
    generator: torch.nn.Module,
    predictor: UTRNoisyPredictor,
    token_embedding: torch.nn.Embedding,
    noise_schedule: dict,
    objective_fn,
    score_mean: float,
    score_std: float,
    device: torch.device,
) -> List[dict]:
    """Run the full guided reverse-diffusion loop for one guidance scale.

    Returns a list of dicts, one per generated sequence, with keys:
        Sequence, Predicted_MRL, HighScore_Prob, GC_Content,
        Max_Homopolymer, Vienna_MFE, Guidance_Scale.
    """
    cfg_gen   = cfg["generation"]
    cfg_diff  = cfg.get("diffusion", {})
    seq_len   = int(cfg_gen["max_len"])
    batch_sz  = int(cfg_gen["batch_size"])
    num_total = int(cfg_gen["num_sequences"])
    t_max     = int(cfg_gen.get("guidance_timestep_max", 200))
    n_steps   = int(cfg_diff.get("num_timesteps", 1000))
    seed      = cfg_gen.get("seed")
    frozen_aug = bool(cfg_gen.get("frozen_aug_codon", False))
    embed_dim  = token_embedding.embedding_dim

    num_batches = math.ceil(num_total / batch_sz)
    results: List[dict] = []

    for b in range(num_batches):
        current_bs = min(batch_sz, num_total - b * batch_sz)

        # ---- T7a: build frozen masks for AUG codon --------------------------
        frozen_mask = frozen_emb = None
        if frozen_aug:
            frozen_mask, frozen_emb = build_frozen_aug_masks(
                current_bs, seq_len, token_embedding, device
            )

        # ---- guided reverse diffusion ----------------------------------------
        batch_seed = None if seed is None else seed + b
        tokens, x_0 = guided_reverse_sample(
            generator=generator,
            predictor=predictor,
            token_embedding=token_embedding,
            noise_schedule=noise_schedule,
            objective_fn=objective_fn,
            batch_size=current_bs,
            seq_len=seq_len,
            embed_dim=embed_dim,
            guidance_scale=scale,
            num_timesteps=n_steps,
            guidance_timestep_max=t_max,
            frozen_mask=frozen_mask,
            frozen_embeddings=frozen_emb,
            seed=batch_seed,
            device=device,
        )

        # ---- score at t=0 ----------------------------------------------------
        scored = score_generated_batch(tokens, predictor, token_embedding, device=device)
        pred_scores = scored["predicted_score"].cpu()
        high_probs  = scored["high_score_prob"].cpu()

        # ---- decode and record metrics ---------------------------------------
        for i in range(current_bs):
            seq_str = decode_tokens(tokens[i].cpu().numpy())
            results.append({
                "Sequence":         seq_str,
                "Predicted_MRL":    pred_scores[i].item() * score_std + score_mean,
                "HighScore_Prob":   high_probs[i].item(),
                "GC_Content":       get_gc_content(seq_str),
                "Max_Homopolymer":  get_max_homopolymer(seq_str),
                "Vienna_MFE":       get_mfe(seq_str),
                "Guidance_Scale":   scale,
            })

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_guided_sequences(config_path: str = "guided_config.yaml") -> None:
    """End-to-end guided UTR generation.

    Reads all settings from ``config_path``, generates sequences, and saves
    one CSV per guidance scale to ``output_dir``.
    """
    cfg     = load_config(config_path)
    cfg_gen = cfg["generation"]
    cfg_diff = cfg.get("diffusion", {})
    cfg_wandb = cfg.get("wandb", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")
    print(f"[generate] Node:   {os.getenv('SLURMD_NODENAME', 'local')}")

    # ---- W&B (optional) -------------------------------------------------
    use_wandb = bool(cfg_wandb.get("enabled", True))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg_wandb.get("project", "utr-guided-inference"),
                name=cfg.get("experiment_name", "utr_guided"),
                config=cfg,
            )
        except Exception as exc:
            print(f"[generate] W&B init failed ({exc}), continuing without it.")
            use_wandb = False

    # ---- Load generator + embedding --------------------------------------
    gen_path = os.path.join(_SCRIPT_DIR, cfg["paths"]["generator_model"])
    print(f"[generate] Loading generator from {gen_path}")
    generator, token_embedding, init_config = load_generator_and_embedding(gen_path, device)

    # ---- Load predictor --------------------------------------------------
    pred_path = os.path.join(_SCRIPT_DIR, cfg["paths"]["noisy_predictor_model"])
    print(f"[generate] Loading predictor from {pred_path}")
    predictor, pred_ckpt = UTRNoisyPredictor.from_checkpoint(pred_path, device)

    score_mean = float(pred_ckpt.get("score_mean", 0.0))
    score_std  = float(pred_ckpt.get("score_std",  1.0))
    print(f"[generate] Score normalisation: mean={score_mean:.4f} std={score_std:.4f}")

    # ---- Build noise schedule --------------------------------------------
    noise_schedule = build_noise_schedule(
        num_timesteps=int(cfg_diff.get("num_timesteps", 1000)),
        beta_start=float(cfg_diff.get("beta_start", 1e-4)),
        beta_end=float(cfg_diff.get("beta_end", 0.02)),
        device=device,
    )

    # ---- Build objective --------------------------------------------------
    objective_fn = build_objective(
        cfg_gen, predictor, token_embedding, score_mean, score_std, device
    )

    # ---- Resolve guidance scales (T7c sweep or single scale) -------------
    sweep_scales: list = cfg_gen.get("guidance_scale_sweep", [])
    if sweep_scales:
        scales = [float(s) for s in sweep_scales]
        print(f"[generate] T7c sweep over guidance scales: {scales}")
    else:
        scales = [float(cfg_gen.get("guidance_scale", 5.0))]
        print(f"[generate] Single guidance scale: {scales[0]}")

    # ---- Output directory ------------------------------------------------
    out_dir = os.path.join(_SCRIPT_DIR, cfg["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # ---- Generate per scale ----------------------------------------------
    all_results: list = []
    scale_summaries: list = []   # one row per scale → diversity_summary.csv

    # Best-of-N config — how many to keep per scale (0 = keep all)
    bon_keep = int(cfg_gen.get("best_of_n_keep", 0))
    if bon_keep > 0:
        print(f"[generate] Best-of-N enabled: will keep top {bon_keep} sequences per scale by Predicted_MRL")

    for w in scales:
        print(f"\n[generate] === guidance_scale = {w} ===")
        results = generate_batch(
            scale=w,
            cfg=cfg,
            generator=generator,
            predictor=predictor,
            token_embedding=token_embedding,
            noise_schedule=noise_schedule,
            objective_fn=objective_fn,
            score_mean=score_mean,
            score_std=score_std,
            device=device,
        )

        # ---- Best-of-N selection --------------------------------------------
        # Generate a large pool (num_sequences), score every sequence with the
        # predictor, then keep only the top-K by Predicted_MRL.  This directly
        # exploits the predictor's discriminative ability at inference time:
        # even a modest predictor can reliably separate the top 10-25% of a
        # large pool, yielding substantially higher mean MRL in the kept set.
        n_generated = len(results)
        df_raw = pd.DataFrame(results)
        raw_mean_mrl  = df_raw["Predicted_MRL"].mean()
        raw_mean_prob = df_raw["HighScore_Prob"].mean()

        if bon_keep > 0 and n_generated > bon_keep:
            df_selected = df_raw.nlargest(bon_keep, "Predicted_MRL").reset_index(drop=True)
            print(
                f"[generate] Best-of-N: selected top {bon_keep}/{n_generated} | "
                f"raw_mean_mrl={raw_mean_mrl:.3f} → "
                f"selected_mean_mrl={df_selected['Predicted_MRL'].mean():.3f}"
            )
        else:
            df_selected = df_raw
            if bon_keep > 0:
                print(
                    f"[generate] Best-of-N: generated {n_generated} ≤ keep={bon_keep}, "
                    "keeping all sequences."
                )

        all_results.extend(df_selected.to_dict("records"))

        scale_tag = f"{w:.1f}".replace(".", "p")

        # Save the full generated pool for analysis
        raw_csv = os.path.join(out_dir, f"optimized_utrs_w{scale_tag}_all.csv")
        df_raw.to_csv(raw_csv, index=False)

        # Save the selected (Best-of-N) sequences — this is the primary output
        csv_path = os.path.join(out_dir, f"optimized_utrs_w{scale_tag}.csv")
        df_selected.to_csv(csv_path, index=False)

        df_scale  = df_selected   # alias for metrics below

        # T6c / T7c: pairwise Hamming diversity over the generated UTR set
        diversity_val = pairwise_hamming_diversity(df_scale["Sequence"].tolist())
        mean_mrl      = df_scale["Predicted_MRL"].mean()
        mean_prob     = df_scale["HighScore_Prob"].mean()
        mean_gc       = df_scale["GC_Content"].mean()
        high_frac     = (df_scale["HighScore_Prob"] > 0.5).mean()

        print(
            f"[generate] Saved {len(df_selected)} UTRs (from pool of {n_generated}) | "
            f"mean_mrl={mean_mrl:.3f} mean_prob={mean_prob:.3f} "
            f"diversity={diversity_val:.4f} → {csv_path}"
        )

        scale_summaries.append({
            "Guidance_Scale":             w,
            "N_Generated":                n_generated,
            "N_Selected":                 len(df_selected),
            "Raw_Mean_Predicted_MRL":     raw_mean_mrl,
            "Raw_Mean_High_Score_Prob":   raw_mean_prob,
            "Mean_Predicted_MRL":         mean_mrl,
            "Mean_High_Score_Prob":       mean_prob,
            "High_Score_Fraction":        high_frac,
            "Mean_GC_Content":            mean_gc,
            "Mean_Max_Homopolymer":       df_scale["Max_Homopolymer"].mean(),
            "Pairwise_Hamming_Diversity": diversity_val,
        })

        if use_wandb:
            import wandb
            wandb.log({
                f"w{scale_tag}/mean_predicted_mrl":        mean_mrl,
                f"w{scale_tag}/raw_mean_predicted_mrl":    raw_mean_mrl,
                f"w{scale_tag}/mean_high_score_prob":      mean_prob,
                f"w{scale_tag}/high_score_fraction":       high_frac,
                f"w{scale_tag}/mean_gc_content":           mean_gc,
                f"w{scale_tag}/mean_max_homopolymer":      df_scale["Max_Homopolymer"].mean(),
                f"w{scale_tag}/pairwise_hamming_diversity": diversity_val,
                f"w{scale_tag}/n_generated":               n_generated,
                f"w{scale_tag}/n_selected":                len(df_selected),
                "guidance_scale": w,
            })

    # ---- Diversity summary CSV (one row per scale, for thesis figures) ---
    df_summary = pd.DataFrame(scale_summaries)
    summary_path = os.path.join(out_dir, "diversity_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[generate] Diversity summary → {summary_path}")

    # ---- Combined CSV (all scales) ---------------------------------------
    if len(scales) > 1:
        df_all = pd.DataFrame(all_results)
        combined_path = os.path.join(out_dir, "optimized_utrs_sweep.csv")
        df_all.to_csv(combined_path, index=False)
        print(f"[generate] Sweep complete. Combined CSV: {combined_path}")

    if use_wandb:
        import wandb
        df_all = pd.DataFrame(all_results)
        wandb.log({
            "final/mean_predicted_mrl":         df_all["Predicted_MRL"].mean(),
            "final/mean_high_score_prob":        df_all["HighScore_Prob"].mean(),
            "final/mean_gc_content":             df_all["GC_Content"].mean(),
            "final/pairwise_hamming_diversity":  scale_summaries[-1]["Pairwise_Hamming_Diversity"],
        })
        wandb.finish()

    print("\n[generate] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimized 5'UTR sequences via predictor-guided DDPM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_SCRIPT_DIR, "guided_config.yaml"),
        help="Path to the guided inference YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_guided_sequences(config_path=args.config)
