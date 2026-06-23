"""Step 2 — Train DAPN adversarial encoder.

Trains the shared encoder E on game-state observations from both CW (source)
and NaSim (target) using gradient reversal so E learns domain-invariant features.
A reconstruction head D keeps latents task-informative.

Input:  data/kc_obs.npz  (from collect_kc_obs.py)
Output: artifacts/models/dapn_encoder_kc7.pt

Usage:
  cd transfer_dapn
  conda run -n cyberwheel python train_dapn_encoder.py
  conda run -n cyberwheel python train_dapn_encoder.py --epochs 300 --wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cyberwheel"))

from envs.kill_chain import LATENT_DIM, CTX_DIM, KC_FEATS, GS_FEATS, ENTRY_SLOT, TARGET_SLOT
from envs.scenario_cfg import get_scenario
from models.dapn_adversarial import DAPNAdversarial, save_encoder

_ROOT      = Path(__file__).resolve().parent
DEFAULT_IN  = str(_ROOT / "data/kc_obs.npz")
DEFAULT_OUT = str(_ROOT / "artifacts/models/dapn_encoder_kc7.pt")

N_STAGES = 5  # 0=pre-entry  1=entry-done  2=target-found  3=target-user  4=target-root


def _infer_layout(kc_ctx: np.ndarray) -> tuple[int, int, int]:
    """Infer (kc_dim, max_slots, game_state_dim) from observation batch."""
    kc_dim = int(kc_ctx.shape[1]) - CTX_DIM
    max_slots = kc_dim // KC_FEATS
    game_state_dim = max_slots * GS_FEATS
    return kc_dim, max_slots, game_state_dim


def _extract_game_state_batch(kc_ctx: np.ndarray, kc_dim: int, max_slots: int) -> np.ndarray:
    kc = kc_ctx[:, :kc_dim]
    return kc.reshape(len(kc), max_slots, KC_FEATS)[:, :, :GS_FEATS].reshape(len(kc), -1)


def _assign_stages(gs: torch.Tensor, max_slots: int) -> torch.Tensor:
    """(N, game_state_dim) → (N,) kill-chain stage 0-4."""
    slots        = gs.reshape(-1, max_slots, GS_FEATS)
    entry_phase  = slots[:, ENTRY_SLOT,  0]
    target_phase = slots[:, TARGET_SLOT, 0]

    stages = torch.zeros(len(gs), dtype=torch.long, device=gs.device)
    stages = torch.where(entry_phase  >= 0.6,                             torch.ones_like(stages),        stages)
    stages = torch.where((entry_phase >= 0.6) & (target_phase  > 0.0) & (target_phase < 0.6),  torch.full_like(stages, 2), stages)
    stages = torch.where((entry_phase >= 0.6) & (target_phase >= 0.6) & (target_phase < 1.0),  torch.full_like(stages, 3), stages)
    stages = torch.where((entry_phase >= 0.6) & (target_phase >= 1.0),    torch.full_like(stages, 4), stages)
    return stages


def _contrastive_loss(latents: torch.Tensor, domain_labels: torch.Tensor,
                      gs: torch.Tensor, max_slots: int) -> torch.Tensor:
    """Pull same-stage CW and NaSim latents to the same mean.

    For each kill-chain stage that appears in both domains, compute the
    per-domain mean latent and minimise the MSE between those means.
    """
    stages   = _assign_stages(gs, max_slots)
    src_mask = (domain_labels == 0)   # CW
    tgt_mask = (domain_labels == 1)   # NaSim

    loss  = torch.tensor(0.0, device=latents.device)
    count = 0
    for stage in range(N_STAGES):
        s_idx = src_mask & (stages == stage)
        t_idx = tgt_mask & (stages == stage)
        if s_idx.sum() < 1 or t_idx.sum() < 1:
            continue
        src_mean = latents[s_idx].mean(dim=0)
        tgt_mean = latents[t_idx].mean(dim=0)
        loss  = loss + F.mse_loss(src_mean, tgt_mean)
        count += 1

    return loss / max(count, 1)


def _alpha_schedule(epoch: int, total: int) -> float:
    """Linearly ramp GRL alpha from 0 → 1 over first half of training."""
    p = min(1.0, 2.0 * epoch / total)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def main():
    ap = argparse.ArgumentParser(description="Train DAPN adversarial encoder")
    ap.add_argument("--scenario", default="two_subnet", choices=["two_subnet", "one_subnet"])
    ap.add_argument("--data",    default=None)
    ap.add_argument("--out",     default=None)
    ap.add_argument("--epochs",  type=int,   default=200)
    ap.add_argument("--batch",   type=int,   default=256)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--lambda-domain", type=float, default=1.0,
                    help="Weight of domain classification loss")
    ap.add_argument("--lambda-recon", type=float, default=1.0,
                    help="Weight of game-state reconstruction loss")
    ap.add_argument("--lambda-contrast", type=float, default=2.0,
                    help="Weight of paired stage-contrastive loss")
    ap.add_argument("--seed",    type=int,   default=42)
    ap.add_argument("--device",  default="cpu")
    ap.add_argument("--wandb",   action="store_true")
    ap.add_argument("--wandb-project", default="nasimemu-exp1")
    args = ap.parse_args()
    cfg = get_scenario(args.scenario)
    if args.data is None:
        args.data = str(_ROOT / cfg.data_kc_obs)
    if args.out is None:
        args.out = str(_ROOT / cfg.encoder_out)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # ── Load data ─────────────────────────────────────────────────────────
    data     = np.load(args.data)
    src_raw  = data["source_obs"].astype(np.float32)
    tgt_raw  = data["target_obs"].astype(np.float32)

    kc_dim, max_slots, game_state_dim = _infer_layout(src_raw)
    assert _infer_layout(tgt_raw)[0] == kc_dim, "source/target KC dim mismatch"

    src_gs = _extract_game_state_batch(src_raw, kc_dim, max_slots)
    tgt_gs = _extract_game_state_batch(tgt_raw, kc_dim, max_slots)

    # Domain labels: 0=CW(source), 1=NaSim(target)
    src_labels = np.zeros(len(src_gs), dtype=np.int64)
    tgt_labels = np.ones( len(tgt_gs), dtype=np.int64)

    all_gs  = np.concatenate([src_gs,     tgt_gs],     axis=0)
    all_lbl = np.concatenate([src_labels, tgt_labels], axis=0)

    ds     = TensorDataset(torch.from_numpy(all_gs), torch.from_numpy(all_lbl))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    print(f"\n{'='*55}")
    print(f"Training DAPN adversarial encoder ({max_slots}-slot KC)")
    print(f"  source         : {len(src_gs)} obs")
    print(f"  target         : {len(tgt_gs)} obs")
    print(f"  kc+ctx dim     : {kc_dim + CTX_DIM}")
    print(f"  game-state dim : {game_state_dim}")
    print(f"  latent dim     : {LATENT_DIM}")
    print(f"  epochs         : {args.epochs}")
    print(f"  λ_domain       : {args.lambda_domain}")
    print(f"  λ_recon        : {args.lambda_recon}")
    print(f"  λ_contrast     : {args.lambda_contrast}")
    print(f"  device         : {device}")
    print(f"{'='*55}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model  = DAPNAdversarial(game_state_dim, LATENT_DIM).to(device)
    opt    = optim.Adam(model.parameters(), lr=args.lr)
    ce     = nn.CrossEntropyLoss()
    mse    = nn.MSELoss()

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name="dapn-encoder",
                config=vars(args))
        except ImportError:
            print("wandb not installed — skipping")

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        alpha = _alpha_schedule(epoch, args.epochs)
        total_loss = total_recon = total_contrast = domain_acc = 0.0
        n_batches  = 0

        for gs_batch, lbl_batch in loader:
            gs_batch  = gs_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            latent, domain_logits, recon = model(gs_batch, alpha=alpha)
            loss_domain   = ce(domain_logits, lbl_batch) * args.lambda_domain
            loss_recon    = mse(recon, gs_batch)         * args.lambda_recon
            loss_contrast = _contrastive_loss(latent, lbl_batch, gs_batch, max_slots) * args.lambda_contrast
            loss          = loss_domain + loss_recon + loss_contrast

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss     += loss.item()
            total_recon    += loss_recon.item()
            total_contrast += loss_contrast.item()
            domain_acc     += (domain_logits.argmax(1) == lbl_batch).float().mean().item()
            n_batches      += 1

        avg_loss     = total_loss     / n_batches
        avg_recon    = total_recon    / n_batches
        avg_contrast = total_contrast / n_batches
        avg_acc      = domain_acc     / n_batches

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  recon={avg_recon:.4f}  "
                  f"contrast={avg_contrast:.4f}  "
                  f"domain_acc={avg_acc:.3f}  alpha={alpha:.3f}")

        if wandb_run:
            wandb_run.log({"loss": avg_loss, "recon_loss": avg_recon,
                           "domain_acc": avg_acc, "alpha": alpha, "epoch": epoch})

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_encoder(model, str(out_path) + ".best.pt",
                         game_state_dim, LATENT_DIM, max_slots, kc_dim)

    save_encoder(model, str(out_path), game_state_dim, LATENT_DIM, max_slots, kc_dim)

    if wandb_run:
        wandb_run.finish()

    print(f"\nFinal encoder → {out_path}")
    print(f"Best encoder  → {out_path}.best.pt")
    print(f"\nNext: python train_cw_dapn_policy.py")


if __name__ == "__main__":
    main()
