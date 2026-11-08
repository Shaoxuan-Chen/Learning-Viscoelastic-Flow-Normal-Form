# ============================================================
# 1. Imports
# ============================================================
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 2. Global reproducibility
# ============================================================
def set_all_seeds(seed=0):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Scale all four variables (matches paper Appendix A)
# ============================================================
def fit_param_scaler(x_np, y_np, r1_np, r2_np, eps=1e-12):
    """
    Max-abs scale all four variables:
        x  = I'
        y  = I''
        r1 = G
        r2 = Wi

    Matches Appendix A: z~ = z / (max_i|z_i| + eps) for z in {x,y,G,Wi}.
    """
    sx = float(np.max(np.abs(x_np)) + eps)
    sy = float(np.max(np.abs(y_np)) + eps)
    sr1 = float(np.max(np.abs(r1_np)) + eps)
    sr2 = float(np.max(np.abs(r2_np)) + eps)
    return dict(sx=sx, sy=sy, sr1=sr1, sr2=sr2)


def apply_param_scaler(x_np, y_np, r1_np, r2_np, scaler):
    """
    Scale all four variables by their fitted max-abs factors.
    """
    x_s = x_np / (scaler["sx"] + 1e-12)
    y_s = y_np / (scaler["sy"] + 1e-12)
    r1_s = r1_np / (scaler["sr1"] + 1e-12)
    r2_s = r2_np / (scaler["sr2"] + 1e-12)
    return x_s, y_s, r1_s, r2_s


def build_denom_vector(exps, scaler):
    """
    For a monomial

        x^px y^py r1^pr1 r2^pr2

    we have
        coef_raw = coef_scaled / (sx^px sy^py sr1^pr1 sr2^pr2)
    """
    sx, sy, sr1, sr2 = scaler["sx"], scaler["sy"], scaler["sr1"], scaler["sr2"]
    denom = np.zeros(len(exps), dtype=np.float64)
    for j, (px, py, pr1, pr2) in enumerate(exps):
        denom[j] = (sx ** px) * (sy ** py) * (sr1 ** pr1) * (sr2 ** pr2)
    return denom


def scaled_to_raw_coefs(coefs_scaled, exps, scaler):
    denom = build_denom_vector(exps, scaler)
    return coefs_scaled / (denom + 1e-12)


def print_scaling_info(df, scaler):
    print("\n[Scaling info: all four variables are max-abs scaled]")
    print(f"G  raw range : [{df['G'].min():.6g}, {df['G'].max():.6g}]")
    print(f"Wi raw range : [{df['Wi'].min():.6g}, {df['Wi'].max():.6g}]")
    print("I' raw range : [{:.6g}, {:.6g}]".format(df["I'"].min(), df["I'"].max()))
    print("I'' raw range: [{:.6g}, {:.6g}]".format(df["I''"].min(), df["I''"].max()))
    print(f"sx  (I'  scale) = {scaler['sx']:.6g}")
    print(f"sy  (I'' scale) = {scaler['sy']:.6g}")
    print(f"sr1 (G scale)   = {scaler['sr1']:.6g}")
    print(f"sr2 (Wi scale)  = {scaler['sr2']:.6g}")
    print(f"G  scaled range : [{df['G'].min()/scaler['sr1']:.6g}, {df['G'].max()/scaler['sr1']:.6g}]")
    print(f"Wi scaled range : [{df['Wi'].min()/scaler['sr2']:.6g}, {df['Wi'].max()/scaler['sr2']:.6g}]")


# ============================================================
# 4. Data augmentation
# ============================================================
def augment_data_flip_y(df, y_col="I''"):
    """
    Duplicate dataset and flip sign of y = I'' in duplicated copy.
    This is symmetry-consistent for the current problem.
    """
    if y_col not in df.columns:
        raise ValueError(f"Column {y_col} not found in DataFrame.")

    df_flip = df.copy()
    df_flip[y_col] = -df_flip[y_col]

    df_aug = pd.concat([df, df_flip], axis=0, ignore_index=True)
    return df_aug


# ============================================================
# 5. Dictionary construction
# ============================================================
def generate_exponents(
    d_xy=3,
    d_r=1,
    d_total=None,
    include_bias=False,
    y_parity=None,
):
    """
    Generate exponent tuples for monomials:
        phi_k = x^px * y^py * r1^pr1 * r2^pr2

    d_xy: maximum total degree in (x, y), i.e. px + py <= d_xy
    d_r : maximum total degree in (r1, r2), i.e. pr1 + pr2 <= d_r
    d_total: optional cap on px + py + pr1 + pr2
    include_bias: whether to include (0, 0, 0, 0)
    y_parity: one of {None, "even", "odd"}
    """
    if y_parity not in {None, "even", "odd"}:
        raise ValueError("y_parity must be one of {None, 'even', 'odd'}")

    exps = []
    for px in range(d_xy + 1):
        for py in range(d_xy + 1 - px):
            deg_xy = px + py

            for pr1 in range(d_r + 1):
                for pr2 in range(d_r + 1 - pr1):
                    deg_r = pr1 + pr2

                    if d_total is not None and (deg_xy + deg_r) > d_total:
                        continue

                    if (px, py, pr1, pr2) == (0, 0, 0, 0) and not include_bias:
                        continue

                    if y_parity == "even" and (py % 2 != 0):
                        continue
                    if y_parity == "odd" and (py % 2 == 0):
                        continue

                    exps.append((px, py, pr1, pr2))
    return exps


def build_feature_matrices(x, y, r1, r2, exps):
    """
    Build Phi, dPhi/dx, dPhi/dy for a given exponent list.
    """
    N = x.shape[0]
    M = len(exps)

    Phi = torch.empty((N, M), dtype=x.dtype, device=x.device)
    dPhi_dx = torch.empty((N, M), dtype=x.dtype, device=x.device)
    dPhi_dy = torch.empty((N, M), dtype=x.dtype, device=x.device)

    for j, (px, py, pr1, pr2) in enumerate(exps):
        phi = (x ** px) * (y ** py) * (r1 ** pr1) * (r2 ** pr2)
        Phi[:, j] = phi

        if px == 0:
            dphi_dx = torch.zeros_like(x)
        else:
            dphi_dx = px * (x ** (px - 1)) * (y ** py) * (r1 ** pr1) * (r2 ** pr2)

        if py == 0:
            dphi_dy = torch.zeros_like(y)
        else:
            dphi_dy = py * (x ** px) * (y ** (py - 1)) * (r1 ** pr1) * (r2 ** pr2)

        dPhi_dx[:, j] = dphi_dx
        dPhi_dy[:, j] = dphi_dy

    return Phi, dPhi_dx, dPhi_dy


# ============================================================
# 6. polynomial-printing helpers
# ============================================================
def format_term(exp):
    px, py, pr1, pr2 = exp
    parts = []
    if px:
        parts.append(f"x^{px}" if px > 1 else "x")
    if py:
        parts.append(f"y^{py}" if py > 1 else "y")
    if pr1:
        parts.append(f"r1^{pr1}" if pr1 > 1 else "r1")
    if pr2:
        parts.append(f"r2^{pr2}" if pr2 > 1 else "r2")
    return "1" if len(parts) == 0 else "*".join(parts)


def top_terms(coefs, exps, k=50, thresh=1e-12):
    # Keep coefficients above thresh, sorted by descending magnitude.
    idx = np.where(np.abs(coefs) > thresh)[0]
    idx = idx[np.argsort(-np.abs(coefs[idx]))]
    out = []
    for j in idx[:k]:
        out.append((coefs[j], format_term(exps[j]), exps[j]))
    return out


# ============================================================
# 7. Core helpers for separate f1 / f2 bases
# ============================================================
def build_all_feature_matrices(x, y, r1, r2, exps_f1, exps_f2):
    # f1 and f2 use separate monomial dictionaries (exps_f1, exps_f2).
    Phi1, dPhi1_dx, dPhi1_dy = build_feature_matrices(x, y, r1, r2, exps_f1)
    Phi2, dPhi2_dx, dPhi2_dy = build_feature_matrices(x, y, r1, r2, exps_f2)
    return {
        "Phi1": Phi1,
        "dPhi1_dx": dPhi1_dx,
        "dPhi1_dy": dPhi1_dy,
        "Phi2": Phi2,
        "dPhi2_dx": dPhi2_dx,
        "dPhi2_dy": dPhi2_dy,
    }


def compute_fJ_from_basis(Phi_pack, a, b):
    """
    Given separate bases for f1 and f2:
        f1 = Phi1 @ a
        f2 = Phi2 @ b
    """
    Phi1 = Phi_pack["Phi1"]
    dPhi1_dx = Phi_pack["dPhi1_dx"]
    dPhi1_dy = Phi_pack["dPhi1_dy"]

    Phi2 = Phi_pack["Phi2"]
    dPhi2_dx = Phi_pack["dPhi2_dx"]
    dPhi2_dy = Phi_pack["dPhi2_dy"]

    f1 = Phi1 @ a
    f2 = Phi2 @ b

    df1_dx = dPhi1_dx @ a
    df1_dy = dPhi1_dy @ a
    df2_dx = dPhi2_dx @ b
    df2_dy = dPhi2_dy @ b

    return f1, f2, df1_dx, df1_dy, df2_dx, df2_dy

#########################################################
# 8. Scaled-space evaluation
#     all four variables (x, y, G, Wi) are max-abs scaled
# ============================================================
@torch.no_grad()
def evaluate_model_scaled(
    df,
    a_scaled,
    b_scaled,
    exps_f1,
    exps_f2,
    scaler,
    alpha,
    beta,
    lambda_stab=1.0,
    lambda_sparse=1e-3,
    # w_det rebalances the loss: tr(J) is linear in the Jacobian
    # entries (O(1e-1) if each entry is O(0.1)), while det(J) is
    # quadratic (O(1e-2)), so without this weight the determinant
    # term would be under-penalized relative to the trace term.
    w_det=100.0,
    device="cpu",
):
    x_np = df["I'"].to_numpy(dtype=np.float64)
    y_np = df["I''"].to_numpy(dtype=np.float64)
    r1_np = df["G"].to_numpy(dtype=np.float64)
    r2_np = df["Wi"].to_numpy(dtype=np.float64)
    x_np, y_np, r1_np, r2_np = apply_param_scaler(x_np, y_np, r1_np, r2_np, scaler)

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    Phi_pack = build_all_feature_matrices(x, y, r1, r2, exps_f1, exps_f2)

    a = torch.tensor(a_scaled, dtype=torch.float64, device=device)
    b = torch.tensor(b_scaled, dtype=torch.float64, device=device)

    f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_basis(Phi_pack, a, b)

    Leq = (f1.pow(2) + f2.pow(2)).mean()

    trJ = df1_dx + df2_dy
    detJ = df1_dx * df2_dy - df1_dy * df2_dx

    Lstab = (F.relu(trJ + alpha).pow(2) + w_det * F.relu(beta - detJ).pow(2)).mean()
    L1 = a.abs().sum() + b.abs().sum()
    loss = Leq + lambda_stab * Lstab + lambda_sparse * L1

    tr_ok = (trJ < -alpha).double().mean()
    det_ok = (detJ > beta).double().mean()
    joint_ok = ((trJ < -alpha) & (detJ > beta)).double().mean()

    return {
        "n": len(df),
        "loss": loss.item(),
        "Leq": Leq.item(),
        "Lstab": Lstab.item(),
        "L1": L1.item(),
        "tr_ok_margin": tr_ok.item(),
        "det_ok_margin": det_ok.item(),
        "joint_ok_margin": joint_ok.item(),
        "trJ_mean": trJ.mean().item(),
        "detJ_mean": detJ.mean().item(),
        "trJ_min": trJ.min().item(),
        "trJ_max": trJ.max().item(),
        "detJ_min": detJ.min().item(),
        "detJ_max": detJ.max().item(),
    }


#########################################################
# 9. Raw-space evaluation
#     Appendix A scales all four inputs (x, y, G, Wi), and the
#     training-time stability margins (alpha, beta) are calibrated
#     against the Jacobian in those SCALED coordinates. So to check
#     the same margin here, the raw-coordinate Jacobian entries must
#     be converted back via the chain rule (d/dx_s = sx * d/dx, etc.)
#     before comparing against alpha/beta.
# ============================================================
@torch.no_grad()
def evaluate_model_raw_exact_margin(
    df,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    scaler,
    alpha,
    beta,
    lambda_stab=1.0,
    lambda_sparse=1e-3,
    w_det=10.0,
    device="cpu",
):
    x_np = df["I'"].to_numpy(dtype=np.float64)
    y_np = df["I''"].to_numpy(dtype=np.float64)
    r1_np = df["G"].to_numpy(dtype=np.float64)
    r2_np = df["Wi"].to_numpy(dtype=np.float64)

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    Phi_pack = build_all_feature_matrices(x, y, r1, r2, exps_f1, exps_f2)

    a = torch.tensor(a_raw, dtype=torch.float64, device=device)
    b = torch.tensor(b_raw, dtype=torch.float64, device=device)

    f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_basis(Phi_pack, a, b)

    Leq = (f1.pow(2) + f2.pow(2)).mean()

    trJ_raw = df1_dx + df2_dy
    detJ_raw = df1_dx * df2_dy - df1_dy * df2_dx

    # Convert the raw-coordinate Jacobian to the scaled-coordinate
    # Jacobian that alpha/beta were calibrated against:
    #   d f1_scaled/d x_s = sx * d f1_raw/d x   (since x_s = x/sx)
    #   d f2_scaled/d y_s = sy * d f2_raw/d y
    # so tr_equiv = sx*f1_x + sy*f2_y, det_equiv = sx*sy*det_raw.
    sx = float(scaler["sx"])
    sy = float(scaler["sy"])
    trJ_equiv = sx * df1_dx + sy * df2_dy
    detJ_equiv = sx * sy * detJ_raw

    Lstab_exact = (
        F.relu(trJ_equiv + alpha).pow(2)
        + w_det * F.relu(beta - detJ_equiv).pow(2)
    ).mean()

    L1 = a.abs().sum() + b.abs().sum()
    loss_exact = Leq + lambda_stab * Lstab_exact + lambda_sparse * L1

    tr_ok_exact = (trJ_equiv < -alpha).double().mean()
    det_ok_exact = (detJ_equiv > beta).double().mean()
    joint_ok_exact = ((trJ_equiv < -alpha) & (detJ_equiv > beta)).double().mean()

    tr_ok_sign = (trJ_raw < 0.0).double().mean()
    det_ok_sign = (detJ_raw > 0.0).double().mean()
    stable_ok_sign = ((trJ_raw < 0.0) & (detJ_raw > 0.0)).double().mean()

    return {
        "n": len(df),
        "loss_exact_margin": loss_exact.item(),
        "Leq": Leq.item(),
        "Lstab_exact_margin": Lstab_exact.item(),
        "L1": L1.item(),
        "sr1": float(scaler["sr1"]),
        "sr2": float(scaler["sr2"]),
        "tr_ok_exact_margin": tr_ok_exact.item(),
        "det_ok_exact_margin": det_ok_exact.item(),
        "joint_ok_exact_margin": joint_ok_exact.item(),
        "tr_ok_sign": tr_ok_sign.item(),
        "det_ok_sign": det_ok_sign.item(),
        "stable_ok_sign": stable_ok_sign.item(),
        "trJ_raw_mean": trJ_raw.mean().item(),
        "trJ_raw_min": trJ_raw.min().item(),
        "trJ_raw_max": trJ_raw.max().item(),
        "detJ_raw_mean": detJ_raw.mean().item(),
        "detJ_raw_min": detJ_raw.min().item(),
        "detJ_raw_max": detJ_raw.max().item(),
        "trJ_equiv_mean": trJ_equiv.mean().item(),
        "trJ_equiv_min": trJ_equiv.min().item(),
        "trJ_equiv_max": trJ_equiv.max().item(),
    }


# ============================================================
# 9.1 Metric summary helpers
# ============================================================
def summarize_scaled_metrics(metrics):
    return {
        "n": metrics["n"],
        "loss": metrics["loss"],
        "Leq": metrics["Leq"],
        "Lstab": metrics["Lstab"],
        "L1": metrics["L1"],
        "tr_ok": metrics["tr_ok_margin"],
        "det_ok": metrics["det_ok_margin"],
        "joint_ok": metrics["joint_ok_margin"],
        "trJ_mean": metrics["trJ_mean"],
        "detJ_mean": metrics["detJ_mean"],
    }


def summarize_raw_exact_metrics(metrics):
    return {
        "n": metrics["n"],
        "loss": metrics["loss_exact_margin"],
        "Leq": metrics["Leq"],
        "Lstab": metrics["Lstab_exact_margin"],
        "L1": metrics["L1"],
        "tr_ok": metrics["tr_ok_exact_margin"],
        "det_ok": metrics["det_ok_exact_margin"],
        "joint_ok": metrics["joint_ok_exact_margin"],
        "trJ_mean": metrics["trJ_equiv_mean"],
        "detJ_mean": metrics["detJ_raw_mean"],
    }


# ============================================================
# 10. Training routine
# ============================================================
def fit_normal_form(
    df_train,
    d_xy=3,
    d_r=1,
    d_total=None,
    include_bias=True,
    lambda_stab=1.0,
    lambda_sparse=1e-3,
    alpha=0.1,
    beta=0.02,
    lr=1e-2,
    w_det=10.0,
    steps=20000,
    lr_milestones=(5000, 12000),
    lr_gamma=0.1,
    standardize=True,
    device="cpu",
    seed=0,
    y_parity_f1="even",
    y_parity_f2="odd",
):
    """
    Learn:
        dx/dt = f1(x,y,r1,r2)  with f1 even in y
        dy/dt = f2(x,y,r1,r2)  with f2 odd in y

    Here:
        x = I'
        y = I''

    All four variables (x, y, r1=G, r2=Wi) are max-abs scaled.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_raw = df_train["I'"].to_numpy(dtype=np.float64)
    y_raw = df_train["I''"].to_numpy(dtype=np.float64)
    r1_raw = df_train["G"].to_numpy(dtype=np.float64)
    r2_raw = df_train["Wi"].to_numpy(dtype=np.float64)

    if standardize:
        scaler = fit_param_scaler(x_raw, y_raw, r1_raw, r2_raw)
        x_np, y_np, r1_np, r2_np = apply_param_scaler(x_raw, y_raw, r1_raw, r2_raw, scaler)
    else:
        scaler = dict(sx=1.0, sy=1.0, sr1=1.0, sr2=1.0)
        x_np, y_np, r1_np, r2_np = x_raw, y_raw, r1_raw, r2_raw

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    exps_f1 = generate_exponents(
        d_xy=d_xy, d_r=d_r, d_total=d_total,
        include_bias=include_bias, y_parity=y_parity_f1,
    )
    exps_f2 = generate_exponents(d_xy=d_xy, d_r=d_r, d_total=d_total, include_bias=include_bias, y_parity=y_parity_f2,)

    Phi_pack = build_all_feature_matrices(x, y, r1, r2, exps_f1, exps_f2)

    M1 = Phi_pack["Phi1"].shape[1]
    M2 = Phi_pack["Phi2"].shape[1]

    a = torch.nn.Parameter(torch.zeros(M1, dtype=torch.float64, device=device))
    b = torch.nn.Parameter(torch.zeros(M2, dtype=torch.float64, device=device))
    # Coefficients start at zero, so Leq is zero at t=0 but Lstab is
    # not (the zero Jacobian fails tr(J) < -alpha, det(J) > beta), so
    # training is initially driven by the stability loss.

    opt = torch.optim.Adam([a, b], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(lr_milestones), gamma=lr_gamma
    )

    history = []

    for t in range(steps):
        opt.zero_grad()
        f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_basis(Phi_pack, a, b)
        
        L_eq = (f1.pow(2) + f2.pow(2)).mean()

        trJ = df1_dx + df2_dy
        detJ = df1_dx * df2_dy - df1_dy * df2_dx

        L_stab = (F.relu(trJ + alpha).pow(2) + w_det * F.relu(beta - detJ).pow(2)).mean()
        L_sparse = a.abs().sum() + b.abs().sum()

        loss = L_eq + lambda_stab * L_stab + lambda_sparse * L_sparse
        loss.backward()
        opt.step()
        scheduler.step()

        if t % 200 == 0 or t == steps - 1:
            with torch.no_grad():
                tr_ok = (trJ < -alpha).double().mean().item()
                det_ok = (detJ > beta).double().mean().item()
                joint_ok = ((trJ < -alpha) & (detJ > beta)).double().mean().item()
                cur_lr = opt.param_groups[0]["lr"]

            history.append(
                (
                    t,
                    cur_lr,
                    loss.item(),
                    L_eq.item(),
                    L_stab.item(),
                    L_sparse.item(),
                    tr_ok,
                    det_ok,
                    joint_ok,
                )
            )

            print(
                f"t={t:5d}  lr={cur_lr:.1e}  loss={loss.item():.3e}  "
                f"Leq={L_eq.item():.3e}  "
                f"Lstab={L_stab.item():.3e}  "
                f"L1={L_sparse.item():.3e}  "
                f"percentage(trJ<-alpha)={tr_ok:.3f}  "
                f"percentage(detJ>beta)={det_ok:.3f}  "
                f"joint={joint_ok:.3f}"
            )
            print(f"(alpha={alpha}, beta={beta})")
    a_scaled = a.detach().cpu().numpy()
    b_scaled = b.detach().cpu().numpy()

    a_raw = scaled_to_raw_coefs(a_scaled, exps_f1, scaler)
    b_raw = scaled_to_raw_coefs(b_scaled, exps_f2, scaler)

    return {
        "a_scaled": a_scaled,
        "b_scaled": b_scaled,
        "a_raw": a_raw,
        "b_raw": b_raw,
        "exps_f1": exps_f1,
        "exps_f2": exps_f2,
        "history": history,
        "scaler": scaler,
        "hyper": dict(
            d_xy=d_xy,
            d_r=d_r,
            d_total=d_total,
            include_bias=include_bias,
            lambda_stab=lambda_stab,
            lambda_sparse=lambda_sparse,
            alpha=alpha,
            beta=beta,
            lr=lr,
            steps=steps,
            w_det=w_det,
            lr_milestones=list(lr_milestones) if lr_milestones is not None else None,
            lr_gamma=lr_gamma,
            standardize=standardize,
            device=device,
            y_parity_f1=y_parity_f1,
            y_parity_f2=y_parity_f2,
        ),
    }


# ============================================================
# 11. Train/test split
# ============================================================
def split_train_test_random(df, train_frac=0.8, seed=0):
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(np.floor(train_frac * n))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    return df_train, df_test


# ============================================================
# 12. Bertini file loading
# ============================================================
 
# Reads a Bertini real_finite_solutions file, formatted as:
#   <n_roots>
#   x1_real x1_imag
#   y1_real y1_imag
#   x2_real x2_imag
#   y2_real y2_imag
#   ...
def load_bertini_real_finite_solutions(filepath, imag_tol=1e-10, keep_complex=False):
    filepath = Path(filepath)

    with filepath.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        raise ValueError(f"Empty file: {filepath}")

    n_roots = int(lines[0])
    data_lines = lines[1:]

    if len(data_lines) != 2 * n_roots:
        raise ValueError(
            f"Unexpected file format in {filepath}. "
            f"Expected {2 * n_roots} data lines after header, got {len(data_lines)}."
        )

    roots = []
    for k in range(n_roots):
        x_line = data_lines[2 * k].split()
        y_line = data_lines[2 * k + 1].split()

        if len(x_line) != 2 or len(y_line) != 2:
            raise ValueError(f"Malformed root block #{k + 1} in {filepath}")

        x_re, x_im = map(float, x_line)
        y_re, y_im = map(float, y_line)

        if keep_complex:
            roots.append([complex(x_re, x_im), complex(y_re, y_im)])
        else:
            if abs(x_im) > imag_tol or abs(y_im) > imag_tol:
                raise ValueError(
                    f"Root #{k + 1} has non-negligible imaginary part: "
                    f"x_im={x_im}, y_im={y_im}"
                )
            roots.append([x_re, y_re])

    return roots


# ============================================================
# 13. Raw-point evaluation
# ============================================================
# Evaluates a raw polynomial model at a point (x, y) = (I', I''):
# f1, f2, the Jacobian, its trace/determinant, the residual norm,
# and the resulting stability sign.
def eval_poly_and_derivs_single_eq(x_val, y_val, G_raw, Wi_raw, coefs, exps):
    f = 0.0
    dfdx = 0.0
    dfdy = 0.0

    for c, (px, py, pr1, pr2) in zip(coefs, exps):
        base = (G_raw ** pr1) * (Wi_raw ** pr2)

        f += c * (x_val ** px) * (y_val ** py) * base

        if px > 0:
            dfdx += c * px * (x_val ** (px - 1)) * (y_val ** py) * base

        if py > 0:
            dfdy += c * (x_val ** px) * py * (y_val ** (py - 1)) * base

    return f, dfdx, dfdy
def eval_fJ_raw_point(
    x_val,
    y_val,
    G_raw,
    Wi_raw,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
):
    f1, df1_dx, df1_dy = eval_poly_and_derivs_single_eq(
        x_val=x_val,
        y_val=y_val,
        G_raw=G_raw,
        Wi_raw=Wi_raw,
        coefs=a_raw,
        exps=exps_f1,
    )

    f2, df2_dx, df2_dy = eval_poly_and_derivs_single_eq(
        x_val=x_val,
        y_val=y_val,
        G_raw=G_raw,
        Wi_raw=Wi_raw,
        coefs=b_raw,
        exps=exps_f2,
    )

    Fval = np.array([f1, f2], dtype=float)
    Jval = np.array([[df1_dx, df1_dy],
                     [df2_dx, df2_dy]], dtype=float)

    return Fval, Jval


def evaluate_point_raw(
    x_val,
    y_val,
    G_raw,
    Wi_raw,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
):
    Fval, Jval = eval_fJ_raw_point(
        x_val=x_val,
        y_val=y_val,
        G_raw=G_raw,
        Wi_raw=Wi_raw,
        a_raw=a_raw,
        b_raw=b_raw,
        exps_f1=exps_f1,
        exps_f2=exps_f2,
    )

    f1_val = float(Fval[0])
    f2_val = float(Fval[1])
    res_norm = float(np.linalg.norm(Fval))

    trJ = float(np.trace(Jval))
    detJ = float(np.linalg.det(Jval))

    return {
        "Iprime": float(x_val),
        "Idoubleprime": float(y_val),
        "Itotal": float(x_val + y_val),
        "F1": f1_val,
        "F2": f2_val,
        "res_norm": res_norm,
        "trJ": trJ,
        "detJ": detJ,
        "stable_sign": (trJ < 0.0) and (detJ > 0.0),
    }

# ============================================================
# 14. State classification
# ============================================================
# Single source of truth for the classification / mirror-pair
# thresholds, so changing one number (e.g. to match a future
# re-derivation from raw data) only requires editing this block.
STATE_MARGIN_X = 0.15
STATE_MARGIN_Y = 0.15

MIRROR_EPS_X = 1e-3
MIRROR_EPS_Y_SUM = 1e-3
MIRROR_EPS_SEP = 1e-3


def classify_state_raw(x_raw, y_raw, margin_x=STATE_MARGIN_X, margin_y=STATE_MARGIN_Y):
    if abs(y_raw) < margin_y:
        if abs(x_raw) < margin_x:
            return "symmetric"
        elif x_raw > margin_x:
            return "D"
        elif x_raw < -margin_x:
            return "C"
        else:
            return "other"
    elif y_raw > margin_y and x_raw > margin_x:
        return "AD+"
    elif y_raw < -margin_y and x_raw > margin_x:
        return "AD-"
    else:
        return "other"

# ============================================================
# 15. Root-level Bertini analysis
# ============================================================
def apply_mirror_pair_correction(
    rows,
    eps_x=MIRROR_EPS_X,
    eps_y_sum=MIRROR_EPS_Y_SUM,
    eps_sep=MIRROR_EPS_SEP,
    margin_x=STATE_MARGIN_X,
    require_in_box=True,
):
    """
    Symmetry-aware label correction (post-processes the pointwise
    threshold labels already assigned in `rows`).

    The pointwise threshold (margin_x, margin_y) answers: "for one
    isolated equilibrium, is the asymmetry large enough to call it
    AD+/AD-?" But the learned vector field has exact reflection
    symmetry (f1 even in y, f2 odd in y), so at a fixed (G, Wi) any
    stable root (x, y) with y != 0 is guaranteed to have a stable
    mirror partner (x, -y). If two stable roots satisfy

        |x1 - x2| < eps_x,   |y1 + y2| < eps_y_sum,   |y1 - y2| > eps_sep

    they are a genuine, well-separated reflection pair (not two
    near-zero numerical duplicates of the same D/symmetric root — the
    eps_sep floor guards against that), and are relabeled AD+/AD- even
    if |y| is still under the pointwise threshold.

    This override only ever applies to a pair whose ORIGINAL threshold
    label was "D" on both sides (equivalently, x > margin_x on both
    sides, since that is exactly what "D" means): only a D-like branch
    bifurcating into an asymmetric pair should ever become AD+/AD-. A
    same-x mirror pair on the C side (x < -margin_x, both labeled "C")
    satisfies the same (x1≈x2, y1≈-y2) geometry but must NOT be
    promoted to AD+/AD- — AD is defined with x > margin_x, so a
    C-branch mirror pair staying "C" is the physically correct
    outcome. Roots that are already AD+/AD-, symmetric, or "other" are
    left untouched either way.

    Must be called on rows that have ALREADY been filtered down to the
    physically-accepted roots (stable_sign, passes_residual_tol, and —
    unless require_in_box=False — in_box). Otherwise a root that is
    only a numerical artifact (large residual, or outside the physical
    box) could still get paired up here, flip its partner's label to
    AD+/AD-, and then be dropped by a later filter — leaving an
    orphaned AD+/AD- whose "partner" was never actually accepted as a
    real root.
    """
    def _accepted(row):
        if not row["stable_sign"] or not row["passes_residual_tol"]:
            return False
        if require_in_box and not row["in_box"]:
            return False
        return True

    def _is_d_branch(row):
        return row["label"] == "D" and row["Iprime"] > margin_x

    n = len(rows)
    used = [False] * n
    for i in range(n):
        if used[i] or not _accepted(rows[i]) or not _is_d_branch(rows[i]):
            continue
        xi, yi = rows[i]["Iprime"], rows[i]["Idoubleprime"]
        for j in range(i + 1, n):
            if used[j] or not _accepted(rows[j]) or not _is_d_branch(rows[j]):
                continue
            xj, yj = rows[j]["Iprime"], rows[j]["Idoubleprime"]
            if (
                abs(xi - xj) < eps_x
                and abs(yi + yj) < eps_y_sum
                and yi * yj < 0
                and abs(yi - yj) > eps_sep
            ):
                if yi > yj:
                    rows[i]["label"] = "AD+"
                    rows[j]["label"] = "AD-"
                else:
                    rows[i]["label"] = "AD-"
                    rows[j]["label"] = "AD+"
                used[i] = True
                used[j] = True
                break
    return rows


def analyze_bertini_roots(
    roots,  # roots to be feed to bertini
    G_raw,
    Wi_raw,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    x_bound=1.5,
    y_bound=1.5,
    margin_x=STATE_MARGIN_X,
    margin_y=STATE_MARGIN_Y,
    residual_tol=1e-7,  #help to sort the bertini roots
):
    # NOTE: this returns the RAW, unfiltered per-point rows (every
    # Bertini root, including unstable / out-of-box / high-residual
    # ones) with only the pointwise threshold label applied. The
    # mirror-pair correction is inter-row and must only run on roots
    # that have survived the physical filters, so it happens in
    # collect_stable_roots_from_bertini() after filtering, not here.
    rows = []

    for k, (x_raw, y_raw) in enumerate(roots, start=1):
        x_raw = float(x_raw)
        y_raw = float(y_raw)

        Fval, Jval = eval_fJ_raw_point(
            x_val=x_raw,
            y_val=y_raw,
            G_raw=G_raw,
            Wi_raw=Wi_raw,
            a_raw=a_raw,
            b_raw=b_raw,
            exps_f1=exps_f1,
            exps_f2=exps_f2,
        )

        trJ = float(np.trace(Jval))
        detJ = float(np.linalg.det(Jval))
        res_norm = float(np.linalg.norm(Fval))
        I_raw = float(x_raw + y_raw)

        in_box = (abs(x_raw) <= x_bound) and (abs(y_raw) <= y_bound)
        stable_sign = (trJ < 0.0) and (detJ > 0.0)
        label = classify_state_raw(x_raw, y_raw, margin_x=margin_x, margin_y=margin_y)

        rows.append({
            "root_id": k,
            "Iprime": x_raw,
            "Idoubleprime": y_raw,
            "Itotal": I_raw,
            "res_norm": res_norm,
            "passes_residual_tol": res_norm < residual_tol,
            "in_box": in_box,
            "trJ": trJ,
            "detJ": detJ,
            "stable_sign": stable_sign,
            "label": label,
        })

    return pd.DataFrame(rows)


# ============================================================
# 16. Detailed pointwise comparison
# ============================================================
def summarize_points_raw(
    points,
    names,
    G_raw,
    Wi_raw,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
):
    # Evaluates every Bertini root, then appends gt_point (if given)
    # as a final "ground_truth" row, so each predicted root can be
    # compared directly against it in the printed table.
    if len(points) != len(names):
        raise ValueError("points and names must have the same length.")

    rows = []
    for name, (x_val, y_val) in zip(names, points):
        out = evaluate_point_raw(
            x_val=x_val,
            y_val=y_val,
            G_raw=G_raw,
            Wi_raw=Wi_raw,
            a_raw=a_raw,
            b_raw=b_raw,
            exps_f1=exps_f1,
            exps_f2=exps_f2,
        )
        out["name"] = name
        rows.append(out)

    cols = [
        "name", "Iprime", "Idoubleprime", "Itotal",
        "F1", "F2", "res_norm", "trJ", "detJ", "stable_sign"
    ]
    return pd.DataFrame(rows)[cols]


def compare_bertini_roots_and_ground_truth(
    bertini_file,
    G_raw,
    Wi_raw,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    gt_point=None,
):
    roots = load_bertini_real_finite_solutions(bertini_file)

    df_roots = analyze_bertini_roots(
        roots=roots,
        G_raw=G_raw,
        Wi_raw=Wi_raw,
        a_raw=a_raw,
        b_raw=b_raw,
        exps_f1=exps_f1,
        exps_f2=exps_f2,
        x_bound=1.5,
        y_bound=1.5,
        margin_x=STATE_MARGIN_X,
        margin_y=STATE_MARGIN_Y,
        residual_tol=1e-7,
    )

    points = list(roots)
    names = [f"bertini_root_{k+1}" for k in range(len(roots))]

    if gt_point is not None:
        points.append(list(gt_point))
        names.append("ground_truth")

    df_compare = summarize_points_raw(
        points=points,
        names=names,
        G_raw=G_raw,
        Wi_raw=Wi_raw,
        a_raw=a_raw,
        b_raw=b_raw,
        exps_f1=exps_f1,
        exps_f2=exps_f2,
    )

    return roots, df_roots, df_compare


# ============================================================
# 17. Automatic Bertini input generation and sweep
# ============================================================
def monomial_to_bertini(px, py, pr1, pr2, G_name="G0", Wi_name="Wi0"):
    parts = []

    if px > 0:
        parts.append("x" if px == 1 else f"x^{px}")
    if py > 0:
        parts.append("y" if py == 1 else f"y^{py}")
    if pr1 > 0:
        parts.append(G_name if pr1 == 1 else f"{G_name}^{pr1}")
    if pr2 > 0:
        parts.append(Wi_name if pr2 == 1 else f"{Wi_name}^{pr2}")

    return "1" if len(parts) == 0 else "*".join(parts)


def build_bertini_polynomial_from_coeffs(
    coefs,
    exps,
    coeff_thresh=1e-10,
    G_name="G0",
    Wi_name="Wi0",
):
    terms = []
    for c, (px, py, pr1, pr2) in zip(coefs, exps):
        if abs(c) <= coeff_thresh:
            continue

        mon = monomial_to_bertini(px, py, pr1, pr2, G_name=G_name, Wi_name=Wi_name)

        if mon == "1":
            terms.append(f"{c:+.12e}")
        else:
            terms.append(f"{c:+.12e}*{mon}")

    return "0" if not terms else "\n    ".join(terms)


def write_bertini_input_from_raw_coeffs(
    input_path,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    G_value,
    Wi_value,
    coeff_thresh=1e-10,
):
    f_poly = build_bertini_polynomial_from_coeffs(a_raw, exps_f1, coeff_thresh=coeff_thresh)
    g_poly = build_bertini_polynomial_from_coeffs(b_raw, exps_f2, coeff_thresh=coeff_thresh)

    content = f"""CONFIG

END;

INPUT

variable_group x, y;
function f, g;

constant G0, Wi0;

G0 = {G_value};
Wi0 = {Wi_value};

f =
    {f_poly}
;

g =
    {g_poly}
;

END;
"""

    input_path = Path(input_path)
    input_path.write_text(content)


def run_bertini(bertini_exec, workdir, input_filename="input"):
    workdir = Path(workdir)
    bertini_exec = Path(bertini_exec)

    if not bertini_exec.exists():
        raise FileNotFoundError(f"Bertini executable not found: {bertini_exec}")
    if not workdir.exists():
        raise FileNotFoundError(f"Bertini workdir not found: {workdir}")
    if not (workdir / input_filename).exists():
        raise FileNotFoundError(f"Input file not found: {workdir / input_filename}")

    return subprocess.run(
        [str(bertini_exec)],
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )


def collect_stable_roots_from_bertini(
    bertini_file,
    G_raw,
    Wi_raw,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    x_bound=1.5,
    y_bound=1.5,
    margin_x=STATE_MARGIN_X,
    margin_y=STATE_MARGIN_Y,
    residual_tol=1e-6,
    require_in_box=True,
):
    roots = load_bertini_real_finite_solutions(bertini_file)

    df_roots = analyze_bertini_roots(
        roots=roots,
        G_raw=G_raw,
        Wi_raw=Wi_raw,
        a_raw=a_raw,
        b_raw=b_raw,
        exps_f1=exps_f1,
        exps_f2=exps_f2,
        x_bound=x_bound,
        y_bound=y_bound,
        margin_x=margin_x,
        margin_y=margin_y,
        residual_tol=residual_tol,
    )

    # Appendix A: roots are retained only when residual < 1e-6.
    if require_in_box:
        df_keep = df_roots[
            (df_roots["stable_sign"] == True)
            & (df_roots["in_box"] == True)
            & (df_roots["passes_residual_tol"] == True)
        ].copy()
    else:
        df_keep = df_roots[
            (df_roots["stable_sign"] == True)
            & (df_roots["passes_residual_tol"] == True)
        ].copy()

    if len(df_keep) == 0:
        return pd.DataFrame(columns=[
            "G", "Wi", "Iprime", "Idoubleprime", "Itotal",
            "trJ", "detJ", "label", "res_norm", "in_box"
        ])

    # Mirror-pair correction runs HERE, only on roots that have already
    # survived the stable_sign / passes_residual_tol / (optionally)
    # in_box filters above — never on raw, not-yet-accepted roots. See
    # apply_mirror_pair_correction()'s docstring for why the ordering
    # matters.
    corrected_rows = apply_mirror_pair_correction(
        df_keep.to_dict("records"),
        require_in_box=require_in_box,
    )
    df_keep = pd.DataFrame(corrected_rows)

    df_keep["G"] = G_raw
    df_keep["Wi"] = Wi_raw

    return df_keep[[
        "G", "Wi", "Iprime", "Idoubleprime", "Itotal",
        "trJ", "detJ", "label", "res_norm", "in_box"
    ]].reset_index(drop=True)


def sweep_bertini_over_parameters(
    G_values,
    Wi_values,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    bertini_exec,
    bertini_workdir,
    coeff_thresh=1e-10,
    save_csv_path=None,
    require_in_box=True,
):
    bertini_workdir = Path(bertini_workdir)
    input_path = bertini_workdir / "input"
    real_roots_path = bertini_workdir / "real_finite_solutions"

    all_rows = []

    for G_val in G_values:
        for Wi_val in Wi_values:
            print(f"\n=== Running Bertini for G={G_val}, Wi={Wi_val} ===")

            write_bertini_input_from_raw_coeffs(
                input_path=input_path,
                a_raw=a_raw,
                b_raw=b_raw,
                exps_f1=exps_f1,
                exps_f2=exps_f2,
                G_value=G_val,
                Wi_value=Wi_val,
                coeff_thresh=coeff_thresh,
            )

            proc = run_bertini(
                bertini_exec=bertini_exec,
                workdir=bertini_workdir,
                input_filename="input",
            )

            if proc.returncode != 0:
                print("Bertini failed.")
                print(proc.stdout)
                print(proc.stderr)
                continue

            if not real_roots_path.exists():
                print(f"real_finite_solutions not found for G={G_val}, Wi={Wi_val}")
                continue

            try:
                df_stable = collect_stable_roots_from_bertini(
                    bertini_file=real_roots_path,
                    G_raw=G_val,
                    Wi_raw=Wi_val,
                    a_raw=a_raw,
                    b_raw=b_raw,
                    exps_f1=exps_f1,
                    exps_f2=exps_f2,
                    require_in_box=require_in_box,
                )
                if len(df_stable) > 0:
                    all_rows.append(df_stable)
                    print(df_stable)
                else:
                    print("No stable roots kept.")
            except Exception as e:
                print(f"Failed to parse/analyze roots for G={G_val}, Wi={Wi_val}: {e}")

    if len(all_rows) == 0:
        df_all = pd.DataFrame(columns=[
            "G", "Wi", "Iprime", "Idoubleprime", "Itotal",
            "trJ", "detJ", "label", "res_norm", "in_box"
        ])
    else:
        df_all = pd.concat(all_rows, ignore_index=True)

    if save_csv_path is not None:
        Path(save_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(save_csv_path, index=False)

    return df_all


# ============================================================
# 18. Plot helpers for bifurcation-style scan
# ============================================================
def scan_wi_and_collect_roots(
    G_fixed,
    Wi_start,
    Wi_end,
    Wi_step,
    a_raw,
    b_raw,
    exps_f1,
    exps_f2,
    bertini_exec,
    bertini_workdir,
    coeff_thresh=1e-10,
    require_in_box=True,
    save_csv_path=None,
):
    Wi_values = list(range(Wi_start, Wi_end + 1, Wi_step))

    df_all = sweep_bertini_over_parameters(
        G_values=[G_fixed],
        Wi_values=Wi_values,
        a_raw=a_raw,
        b_raw=b_raw,
        exps_f1=exps_f1,
        exps_f2=exps_f2,
        bertini_exec=bertini_exec,
        bertini_workdir=bertini_workdir,
        coeff_thresh=coeff_thresh,
        require_in_box=require_in_box,
        save_csv_path=save_csv_path,
    )

    return df_all


def make_combined_bifurcation_plot(
    df_scan_by_G,
    G_order,
    df_gt=None,
    g_tol=1e-3,
    save_path=None,
):
    """
    Combine the bifurcation diagrams for several G values into a single
    figure: one column per G value, three rows (I', I'', I = I'+I'').

    df_scan_by_G: dict mapping G_fixed -> df_roots (as returned by
        scan_wi_and_collect_roots).
    G_order: sequence of G values giving the left-to-right column order.

    If df_gt is given (the original, non-augmented experimental
    observations), overlay the matching G-slice in each column as
    black open circles ("GT"), matching the paper's figure style.
    """
    # Marker and color are both fixed per state (not derived from
    # matplotlib's per-axes color cycle), matching the phase-diagram
    # convention and Hopkins et al. Fig. 3's AD+/AD-/C symbols (red
    # up-triangle, blue down-triangle, orange square).
    style_map = {
        "symmetric": {"marker": "x", "color": "black"},
        # D uses tab:green here (not blue) to avoid ambiguity with AD-,
        # which the bifurcation diagram shows as a separate branch.
        # The phase-diagram style_map keeps D=blue, matching Hopkins
        # et al.; the two figures serve different purposes and are
        # allowed to use different colors for the same state.
        "D": {"marker": "<", "color": "#2ca02c"},
        "C": {"marker": "s", "color": "orange"},
        "AD+": {"marker": "^", "color": "red"},
        "AD-": {"marker": "v", "color": "blue"},
        "other": {"marker": ".", "color": "gray"},
    }

    pred_markersize = 9
    gt_markersize = 8
    label_fontsize = 22
    tick_fontsize = 18
    legend_fontsize = 16

    n_rows = 3
    n_cols = len(G_order)
    # Each individual subplot has a width:height ratio of 8:4.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), sharex="col"
    )
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    # Track every label seen across all columns so the legend can be
    # built once, independent of which column happens to contain it.
    all_labels_present = set()

    for col, G_fixed in enumerate(G_order):
        ax0, ax1, ax2 = axes[0, col], axes[1, col], axes[2, col]

        df_roots = df_scan_by_G.get(G_fixed)
        if df_roots is None or df_roots.empty:
            print(f"No roots available for plotting at G={G_fixed}.")
        else:
            df_plot = df_roots.copy()
            df_plot["Itotal"] = df_plot["Iprime"] + df_plot["Idoubleprime"]
            labels_present = sorted(df_plot["label"].unique())
            all_labels_present.update(labels_present)

            for lab in labels_present:
                dfi = df_plot[df_plot["label"] == lab]
                style = style_map.get(lab, {"marker": "o", "color": "black"})
                ax0.plot(
                    dfi["Wi"].to_numpy(), dfi["Iprime"].to_numpy(),
                    linestyle="None", marker=style["marker"],
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.3,
                    markersize=pred_markersize, label=f"pred {lab}",
                )
            for lab in labels_present:
                dfi = df_plot[df_plot["label"] == lab]
                style = style_map.get(lab, {"marker": "o", "color": "black"})
                ax1.plot(
                    dfi["Wi"].to_numpy(), dfi["Idoubleprime"].to_numpy(),
                    linestyle="None", marker=style["marker"],
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.3,
                    markersize=pred_markersize,
                )
            for lab in labels_present:
                dfi = df_plot[df_plot["label"] == lab]
                style = style_map.get(lab, {"marker": "o", "color": "black"})
                ax2.plot(
                    dfi["Wi"].to_numpy(), dfi["Itotal"].to_numpy(),
                    linestyle="None", marker=style["marker"],
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.3,
                    markersize=pred_markersize,
                )

        # Ground-truth overlay (experimental observations), shown as
        # black open circles.
        if df_gt is not None:
            df_gt_g = df_gt[np.isclose(df_gt["G"], G_fixed, atol=g_tol)].copy()

            if len(df_gt_g) > 0:
                df_gt_g["Itotal"] = df_gt_g["I'"] + df_gt_g["I''"]

                ax0.plot(
                    df_gt_g["Wi"], df_gt_g["I'"],
                    linestyle="None", marker="o", markersize=gt_markersize,
                    markerfacecolor="none", markeredgecolor="black",
                    label="GT",
                )
                ax1.plot(
                    df_gt_g["Wi"], df_gt_g["I''"],
                    linestyle="None", marker="o", markersize=gt_markersize,
                    markerfacecolor="none", markeredgecolor="black",
                )
                ax2.plot(
                    df_gt_g["Wi"], df_gt_g["Itotal"],
                    linestyle="None", marker="o", markersize=gt_markersize,
                    markerfacecolor="none", markeredgecolor="black",
                )
            else:
                print(f"No GT points found for G={G_fixed} with g_tol={g_tol}")

        ax0.axhline(0.0, linestyle="--", linewidth=1)
        ax1.axhline(0.0, linestyle="--", linewidth=1)
        ax2.axhline(0.0, linestyle="--", linewidth=1)

        ax0.set_title(rf"$G = {G_fixed:.3f}$", fontsize=label_fontsize)
        ax2.set_xlabel(r"$Wi$", fontsize=label_fontsize)

        for ax in (ax0, ax1, ax2):
            ax.tick_params(axis="both", labelsize=tick_fontsize)

        if col == 0:
            ax0.set_ylabel(r"$I'$", fontsize=label_fontsize)
            ax1.set_ylabel(r"$I''$", fontsize=label_fontsize)
            ax2.set_ylabel(r"$I = I' + I''$", fontsize=label_fontsize)

    # Build the legend from a fixed set of proxy handles covering
    # every state seen anywhere in the figure, in a stable order,
    # rather than from whatever happens to be in the last column.
    legend_handles = []
    for lab in style_map:
        if lab in all_labels_present:
            style = style_map[lab]
            legend_handles.append(
                Line2D(
                    [0], [0], linestyle="None",
                    marker=style["marker"],
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.3,
                    markersize=8, label=f"pred {lab}",
                )
            )
    if df_gt is not None:
        legend_handles.append(
            Line2D(
                [0], [0], linestyle="None", marker="o",
                markerfacecolor="none", markeredgecolor="black", color="black",
                markersize=8, label="GT",
            )
        )

    if len(legend_handles) > 0:
        axes[0, n_cols - 1].legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.90),
            fontsize=legend_fontsize,
            markerscale=0.9,
            handletextpad=0.4,
            labelspacing=0.3,
            borderpad=0.4,
            framealpha=1.0,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        # Also save an EPS (vector) version alongside the PNG.
        eps_save_path = Path(save_path).with_suffix(".eps")
        fig.savefig(
            eps_save_path,
            format="eps",
            dpi=600,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# 18.1 Phase-diagram helpers
# ============================================================
def get_combined_state(labels):
    simple = set()
    for lab in labels:
        if lab in ["AD+", "AD-"]:
            simple.add("AD")
        elif lab in ["D", "C", "symmetric"]:
            simple.add(lab)

    if "AD" in simple and "C" in simple:
        return "ADvC"
    if "D" in simple and "C" in simple:
        return "DvC"
    if "AD" in simple:
        return "AD"
    if "C" in simple:
        return "C"
    if "D" in simple:
        return "D"
    if "symmetric" in simple:
        return "symmetric"
    return "other"


def generate_phase_diagram_data(df_all_roots, G_grid, Wi_grid):
    rows = []

    if df_all_roots.empty or "G" not in df_all_roots.columns:
        for G_val in G_grid:
            for Wi_val in Wi_grid:
                rows.append({
                    "G": G_val,
                    "Wi": Wi_val,
                    "phase": "other",
                    "labels": "",
                })
        return pd.DataFrame(rows)

    for G_val in G_grid:
        for Wi_val in Wi_grid:
            group = df_all_roots[
                np.isclose(df_all_roots["G"], G_val)
                & np.isclose(df_all_roots["Wi"], Wi_val)
            ].copy()

            if "in_box" in group.columns:
                group = group[group["in_box"] == True]

            if len(group) == 0:
                phase = "other"
                labels = []
            else:
                labels = sorted(group["label"].unique().tolist())
                phase = get_combined_state(labels)

            rows.append({
                "G": G_val,
                "Wi": Wi_val,
                "phase": phase,
                "labels": ",".join(labels),
            })

    return pd.DataFrame(rows)


def plot_figure4_phase_diagram(df_phase, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    style_map = {
        "symmetric": dict(marker="x", color="black", label="symmetric", s=80),
        "D": dict(marker="<", color="blue", label="D", s=90),
        "C": dict(marker="s", color="orange", label="C", s=80),
        "AD": dict(marker="D", color="red", label="AD", s=80),
        "DvC": dict(marker="o", facecolors="white", edgecolors="black", label="D v C", s=110),
        "ADvC": dict(marker="*", color="magenta", label="AD v C", s=180),
        "other": dict(marker=".", color="gray", label="other", s=40),
    }

    for phase, style in style_map.items():
        sub = df_phase[df_phase["phase"] == phase]
        if sub.empty:
            continue

        kwargs = style.copy()
        label = kwargs.pop("label")

        ax.scatter(sub["G"], sub["Wi"], label=label, zorder=3, **kwargs)

    ax.set_yscale("log")
    ax.set_xlabel(r"$G$", fontsize=14)
    ax.set_ylabel(r"$Wi$", fontsize=14)
    # EPS does not support transparency, so the grid uses a solid
    # light-gray color instead of alpha, and the legend frame is opaque.
    ax.grid(True, which="both", linestyle="--", color="lightgray")
    ax.legend(loc="lower left", fontsize=11, framealpha=1.0)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        eps_save_path = Path(save_path).with_suffix(".eps")
        fig.savefig(eps_save_path, format="eps", dpi=600, bbox_inches="tight")

    plt.show()


# ============================================================
# 19. Main script
# ============================================================
if __name__ == "__main__":
    # --------------------------------------------------------
    # 19.1 Seeds
    # --------------------------------------------------------
    GLOBAL_SEED = 7
    set_all_seeds(GLOBAL_SEED)

    # --------------------------------------------------------
    # 19.2 Load data
    # Appendix A: the final model is fit using all available
    # equilibrium observations, with no train/test split.
    # --------------------------------------------------------
    df = pd.read_excel("data/data_flow.xlsx")
    df_train = df.copy()

    # Ablation: no reflected-data augmentation, trained on the
    # original (non-mirrored) observations only.
    # df_train = augment_data_flip_y(df_train, y_col="I''")

    print(f"Full dataset size (all equilibrium observations): {len(df)}")
    print(f"Train size (no augmentation): {len(df_train)}")

    # --------------------------------------------------------
    # 19.3 Train model
    # --------------------------------------------------------
    result = fit_normal_form(
        df_train,
        d_xy=4,
        d_r=2,
        d_total=5,
        include_bias=False,
        lambda_stab=500,
        lambda_sparse=1e-5,
        alpha=0.03,
        beta=1e-5,
        lr=5e-4,
        steps=25000,
        lr_milestones=(12000, 20000),
        lr_gamma=0.2,
        w_det=100.0,
        standardize=True,
        device="cpu",
        seed=GLOBAL_SEED,
        # [SYMMETRY-RESTRICTED] Reflection-symmetry constraint: f1 even
        # in y, f2 odd in y, still without reflected-data augmentation.
        y_parity_f1="even",
        y_parity_f2="odd",
    )

    h = result["hyper"]
    print_scaling_info(df_train, result["scaler"])

    # --------------------------------------------------------
    # 19.4 Evaluate model
    # The paper does not hold out a separate test set for this model,
    # so metrics are reported on the fitted (training) observations.
    # --------------------------------------------------------
    train_metrics_scaled = evaluate_model_scaled(
        df_train,
        a_scaled=result["a_scaled"],
        b_scaled=result["b_scaled"],
        exps_f1=result["exps_f1"],
        exps_f2=result["exps_f2"],
        scaler=result["scaler"],
        alpha=h["alpha"],
        beta=h["beta"],
        lambda_stab=h["lambda_stab"],
        lambda_sparse=h["lambda_sparse"],
        w_det=h["w_det"],
        device=h["device"],
    )

    print("\n[Scaled-space metrics]")
    print("metrics:", summarize_scaled_metrics(train_metrics_scaled))

    train_metrics_raw_exact = evaluate_model_raw_exact_margin(
        df_train,
        a_raw=result["a_raw"],
        b_raw=result["b_raw"],
        exps_f1=result["exps_f1"],
        exps_f2=result["exps_f2"],
        scaler=result["scaler"],
        alpha=h["alpha"],
        beta=h["beta"],
        lambda_stab=h["lambda_stab"],
        lambda_sparse=h["lambda_sparse"],
        w_det=h["w_det"],
        device=h["device"],
    )

    print("\n[Raw-space metrics]")
    print("metrics:", summarize_raw_exact_metrics(train_metrics_raw_exact))

    # --------------------------------------------------------
    # 19.5 Print learned terms
    # --------------------------------------------------------
    print("\nTOP TERMS for f1 (scaled coefficients):")
    for c, s, _ in top_terms(result["a_scaled"], result["exps_f1"], k=50, thresh=0):
        print(f"{c:+.3e} * {s}")

    print("\nTOP TERMS for f2 (scaled coefficients):")
    for c, s, _ in top_terms(result["b_scaled"], result["exps_f2"], k=50, thresh=0):
        print(f"{c:+.3e} * {s}")

    print("\nTOP TERMS for f1 (raw coefficients):")
    for c, s, _ in top_terms(result["a_raw"], result["exps_f1"], k=50, thresh=0):
        print(f"{c:+.3e} * {s}")

    print("\nTOP TERMS for f2 (raw coefficients):")
    for c, s, _ in top_terms(result["b_raw"], result["exps_f2"], k=50, thresh=0):
        print(f"{c:+.3e} * {s}")


    # --------------------------------------------------------
    # 19.6 Scan Wi = 1,6,11,...,196 at fixed G
    # --------------------------------------------------------
    # Edit these two paths to point to your own Bertini installation.
    bertini_exec = "/path/to/BertiniApple_v1.7/bertini"
    bertini_workdir = Path("/path/to/BertiniApple_v1.7/examples/viscoelastic_flow")

    # Each script gets its own subfolder so outputs don't mix together.
    figure_dir = Path("paper_figures") / Path(__file__).stem
    figure_dir.mkdir(parents=True, exist_ok=True)

    G_list = [0.5, 0.588, 0.603]
    df_scan_by_G = {}

    for G_fixed in G_list:
        df_scan = scan_wi_and_collect_roots(
            G_fixed=G_fixed,
            Wi_start=1,
            Wi_end=200,
            Wi_step=5,
            a_raw=result["a_raw"],
            b_raw=result["b_raw"],
            exps_f1=result["exps_f1"],
            exps_f2=result["exps_f2"],
            bertini_exec=bertini_exec,
            bertini_workdir=bertini_workdir,
            coeff_thresh=1e-10,
            require_in_box=True,
            save_csv_path=None,
        )

        print(f"\n[Scanned stable roots for G={G_fixed}]")
        print(df_scan.head())
        print(f"Total kept roots: {len(df_scan)}")

        df_scan_by_G[G_fixed] = df_scan

    fig_save_path = figure_dir / "polynomial_evenodd_no_augmentation.png"
    make_combined_bifurcation_plot(
        df_scan_by_G=df_scan_by_G,
        G_order=G_list,
        df_gt=df,          # original non-augmented experimental data
        save_path=fig_save_path,
    )
    print(f"Saved combined figure to: {fig_save_path}")

    # --------------------------------------------------------
    # 19.7 Phase diagram (G, Wi) sweep.
    # G=0.50,0.51,...,0.63 (G_STEP=0.01), Wi=8,13,...,198.
    # --------------------------------------------------------
    G_STEP = 0.01
    WI_STEP = 5
    G_grid = np.round(np.arange(0.50, 0.63 + 1e-9, G_STEP), 3).tolist()
    Wi_grid = list(range(8, 198 + 1, WI_STEP))

    phase_roots_csv = figure_dir / "polynomial_evenodd_no_augmentation_phase_roots.csv"
    phase_summary_csv = figure_dir / "polynomial_evenodd_no_augmentation_phase_summary.csv"
    phase_fig_path = figure_dir / "polynomial_evenodd_no_augmentation_phase_diagram.png"

    # The (G, Wi) sweep is expensive (~1000 Bertini calls). If a
    # previous run already saved results, load them instead of
    # recomputing. Delete the CSVs (or set FORCE_RECOMPUTE_PHASE=True)
    # to force a fresh sweep.
    FORCE_RECOMPUTE_PHASE = False

    if (
        not FORCE_RECOMPUTE_PHASE
        and phase_roots_csv.exists()
        and phase_summary_csv.exists()
    ):
        print(f"\n[Phase diagram] Loading cached results from:")
        print(f"  {phase_roots_csv}")
        print(f"  {phase_summary_csv}")
        df_all_grid = pd.read_csv(phase_roots_csv)
        df_phase = pd.read_csv(phase_summary_csv)
    else:
        print(f"\n[Phase diagram] G_grid ({len(G_grid)} values): {G_grid}")
        print(f"[Phase diagram] Wi_grid ({len(Wi_grid)} values): {Wi_grid}")

        all_phase_rows = []
        for G_val in G_grid:
            df_g = scan_wi_and_collect_roots(
                G_fixed=G_val,
                Wi_start=min(Wi_grid),
                Wi_end=max(Wi_grid),
                Wi_step=WI_STEP,
                a_raw=result["a_raw"],
                b_raw=result["b_raw"],
                exps_f1=result["exps_f1"],
                exps_f2=result["exps_f2"],
                bertini_exec=bertini_exec,
                bertini_workdir=bertini_workdir,
                coeff_thresh=1e-10,
                require_in_box=True,
                save_csv_path=None,
            )
            if len(df_g) > 0:
                all_phase_rows.append(df_g)

        df_all_grid = pd.concat(all_phase_rows, ignore_index=True) if all_phase_rows else pd.DataFrame()

        df_phase = generate_phase_diagram_data(
            df_all_roots=df_all_grid,
            G_grid=G_grid,
            Wi_grid=Wi_grid,
        )

        df_all_grid.to_csv(phase_roots_csv, index=False)
        df_phase.to_csv(phase_summary_csv, index=False)

    print("\n[Phase counts]")
    print(df_phase["phase"].value_counts())

    plot_figure4_phase_diagram(
        df_phase=df_phase,
        save_path=phase_fig_path,
    )
    print(f"Saved phase diagram to: {phase_fig_path}")
    
