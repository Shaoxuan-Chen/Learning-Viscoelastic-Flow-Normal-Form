# nn_normal_form.py
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import root
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# 0. Data Flip
# ============================================================
def augment_data_flip_y(df, y_col="I''"):

    df_flip = df.copy()

    df_flip[y_col] = -df_flip[y_col]

    return pd.concat([df, df_flip], axis=0, ignore_index=True)


# ============================================================
# 1. Reproducibility
# ============================================================
def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 2. Scale ONLY G and Wi
# ============================================================
def fit_param_scaler(r1_np, r2_np, eps=1e-12):
    return {
        "sr1": float(np.max(np.abs(r1_np)) + eps),
        "sr2": float(np.max(np.abs(r2_np)) + eps),
    }


def apply_param_scaler(x_np, y_np, r1_np, r2_np, scaler):
    x_s = x_np.copy()
    y_s = y_np.copy()
    r1_s = r1_np / scaler["sr1"]
    r2_s = r2_np / scaler["sr2"]
    return x_s, y_s, r1_s, r2_s


# ============================================================
# 3. Train/test split
# ============================================================
def split_train_test_random(df, train_frac=0.8, seed=0):
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(np.floor(train_frac * n))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


# ============================================================
# 4. Neural network model: averaging hard constraint
# ============================================================
class DynamicalSystemNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        # raw input is [x, y, G, Wi]
        self.raw_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def raw_forward(self, x, y, r1, r2):
        inputs = torch.stack([x, y, r1, r2], dim=1)
        out = self.raw_net(inputs)

        g1 = out[:, 0]
        g2 = out[:, 1]

        return g1, g2

    def forward(self, x, y, r1, r2):
        # raw NN output at y
        g1_pos, g2_pos = self.raw_forward(x, y, r1, r2)

        # raw NN output at -y
        g1_neg, g2_neg = self.raw_forward(x, -y, r1, r2)

        # enforce f1 even in y
        f1 = 0.5 * (g1_pos + g1_neg)

        # enforce f2 odd in y
        f2 = 0.5 * (g2_pos - g2_neg)

        return f1, f2


# ============================================================
# 5. Compute f and Jacobian by autograd
# ============================================================
def compute_fJ_from_nn(model, x, y, r1, r2):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)

    f1, f2 = model(x, y, r1, r2)

    df1_dx = torch.autograd.grad(
        f1, x, grad_outputs=torch.ones_like(f1), create_graph=True
    )[0]
    df1_dy = torch.autograd.grad(
        f1, y, grad_outputs=torch.ones_like(f1), create_graph=True
    )[0]

    df2_dx = torch.autograd.grad(
        f2, x, grad_outputs=torch.ones_like(f2), create_graph=True
    )[0]
    df2_dy = torch.autograd.grad(
        f2, y, grad_outputs=torch.ones_like(f2), create_graph=True
    )[0]

    return f1, f2, df1_dx, df1_dy, df2_dx, df2_dy


# ============================================================
# 5.1 Leading real part of the Jacobian eigenvalues (for the
#     negative-root fine-tuning penalty below)
# ============================================================
def max_real_eig_2x2(df1_dx, df1_dy, df2_dx, df2_dy):

    trJ = df1_dx + df2_dy

    detJ = (
        df1_dx * df2_dy
        - df1_dy * df2_dx
    )

    delta = trJ.pow(2) - 4.0 * detJ

    sqrt_delta = torch.sqrt(
        torch.clamp(delta, min=0.0) + 1e-12)

    lambda_real_case = 0.5 * (
        trJ + sqrt_delta)

    lambda_complex_case = 0.5 * trJ

    lambda_max_real = torch.where(
        delta >= 0,
        lambda_real_case,
        lambda_complex_case,)

    return lambda_max_real


def make_neg_tensors_from_df(df, scaler, device):
    x_raw = df["Iprime"].to_numpy(dtype=np.float64)
    y_raw = df["Idoubleprime"].to_numpy(dtype=np.float64)
    G_raw = df["G"].to_numpy(dtype=np.float64)
    Wi_raw = df["Wi"].to_numpy(dtype=np.float64)

    x_np, y_np, r1_np, r2_np = apply_param_scaler(
        x_raw, y_raw, G_raw, Wi_raw, scaler
    )

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    return x, y, r1, r2


# ============================================================
# 6. Training
# ============================================================
def fit_normal_form_nn(
    df_train,
    hidden_dim=128,
    lambda_stab=50.0,
    alpha=0.09,
    beta=1e-3,
    w_det=50.0,
    lr=5e-4,
    weight_decay=1e-5,
    steps=60000,
    lr_milestones=(20000, 40000),
    lr_gamma=0.1,
    standardize=True,
    device="cpu",
    seed=7,
):
    set_all_seeds(seed)

    x_raw = df_train["I'"].to_numpy(dtype=np.float64)
    y_raw = df_train["I''"].to_numpy(dtype=np.float64)
    r1_raw = df_train["G"].to_numpy(dtype=np.float64)
    r2_raw = df_train["Wi"].to_numpy(dtype=np.float64)

    if standardize:
        scaler = fit_param_scaler(r1_raw, r2_raw)
    else:
        scaler = {"sr1": 1.0, "sr2": 1.0}

    x_np, y_np, r1_np, r2_np = apply_param_scaler(
        x_raw, y_raw, r1_raw, r2_raw, scaler)

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    model = DynamicalSystemNN(hidden_dim=hidden_dim).to(device).double()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt,
        milestones=list(lr_milestones),
        gamma=lr_gamma,
    )

    history = []

    for t in range(steps):
        opt.zero_grad()

        f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_nn(
            model, x, y, r1, r2
        )

        L_eq = (f1.pow(2) + f2.pow(2)).mean()

        trJ = df1_dx + df2_dy
        detJ = df1_dx * df2_dy - df1_dy * df2_dx

        L_stab = (
            F.relu(trJ + alpha).pow(2)
            + w_det * F.relu(beta - detJ).pow(2)
        ).mean()

        loss = L_eq + lambda_stab * L_stab

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
                {
                    "step": t,
                    "lr": cur_lr,
                    "loss": loss.item(),
                    "Leq": L_eq.item(),
                    "Lstab": L_stab.item(),
                    "tr_ok": tr_ok,
                    "det_ok": det_ok,
                    "joint_ok": joint_ok,
                }
            )

            print(
                f"t={t:6d} lr={cur_lr:.1e} "
                f"loss={loss.item():.3e} "
                f"Leq={L_eq.item():.3e} "
                f"Lstab={L_stab.item():.3e} "
                f"tr_ok={tr_ok:.3f} "
                f"det_ok={det_ok:.3f} "
                f"joint={joint_ok:.3f}"
            )

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "hyper": {
            "hidden_dim": hidden_dim,
            "lambda_stab": lambda_stab,
            "alpha": alpha,
            "beta": beta,
            "w_det": w_det,
            "lr": lr,
            "weight_decay": weight_decay,
            "steps": steps,
            "standardize": standardize,
            "device": device,
        },
    }


# ============================================================
# 6.1 Negative-root fine-tuning: penalize specific (x,y,G,Wi)
#     points for being spuriously stable, without disturbing the
#     main equilibrium/stability loss on the real data.
# ============================================================
def fine_tune_with_negative_roots_nn(
    result,
    df_train,
    df_neg,
    lambda_stab=50.0,
    lambda_neg=50.0,
    gamma_neg=0.01,
    alpha=0.09,
    beta=1e-3,
    w_det=50.0,
    lr=1e-4,
    weight_decay=0.0,
    steps=10000,
    device="cpu",
):
    model = result["model"]
    scaler = result["scaler"]

    x_raw = df_train["I'"].to_numpy(dtype=np.float64)
    y_raw = df_train["I''"].to_numpy(dtype=np.float64)
    r1_raw = df_train["G"].to_numpy(dtype=np.float64)
    r2_raw = df_train["Wi"].to_numpy(dtype=np.float64)

    x_np, y_np, r1_np, r2_np = apply_param_scaler(
        x_raw, y_raw, r1_raw, r2_raw, scaler
    )

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    x_neg, y_neg, r1_neg, r2_neg = make_neg_tensors_from_df(
        df_neg, scaler=scaler, device=device
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    history = []

    for t in range(steps):
        opt.zero_grad()

        # positive/original training loss
        f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_nn(
            model, x, y, r1, r2
        )

        L_eq = (f1.pow(2) + f2.pow(2)).mean()

        trJ = df1_dx + df2_dy
        detJ = df1_dx * df2_dy - df1_dy * df2_dx

        L_stab = (
            F.relu(trJ + alpha).pow(2)
            + w_det * F.relu(beta - detJ).pow(2)
        ).mean()

        # negative roots loss
        _, _, n_df1_dx, n_df1_dy, n_df2_dx, n_df2_dy = compute_fJ_from_nn(
            model, x_neg, y_neg, r1_neg, r2_neg
        )

        lambda_max_real = max_real_eig_2x2(
            n_df1_dx, n_df1_dy, n_df2_dx, n_df2_dy
        )

        L_neg = F.relu(gamma_neg - lambda_max_real).pow(2).mean()

        loss = L_eq + lambda_stab * L_stab + lambda_neg * L_neg

        loss.backward()
        opt.step()

        if t % 200 == 0 or t == steps - 1:
            neg_unstable_ok = (lambda_max_real > gamma_neg).double().mean().item()

            print(
                f"[fine-tune] t={t:6d} "
                f"loss={loss.item():.3e} "
                f"Leq={L_eq.item():.3e} "
                f"Lstab={L_stab.item():.3e} "
                f"Lneg={L_neg.item():.3e} "
                f"neg_ok={neg_unstable_ok:.3f}"
            )

            history.append({
                "step": t,
                "loss": loss.item(),
                "Leq": L_eq.item(),
                "Lstab": L_stab.item(),
                "Lneg": L_neg.item(),
                "neg_ok": neg_unstable_ok,
            })

    result["history_neg"] = result.get("history_neg", []) + history

    return result


# ============================================================
# 6.2 Collect spurious stable C roots from the "pink region"
#     (a specific (G,Wi) sub-region where the base-trained model
#     tends to predict wrong stable C branches)
# ============================================================
def collect_negative_c_roots_from_pink_region(
    result,
    G_min=0.50,
    G_max=0.55,
    G_step=0.01,
    Wi_min=100,
    Wi_max=200,
    Wi_step=5,
    require_in_box=False,
):
    all_neg = []

    G_grid = np.round(np.arange(G_min, G_max + 1e-12, G_step), 3)

    for G_val in G_grid:
        for Wi_val in range(Wi_min, Wi_max + 1, Wi_step):
            df_roots = find_roots_nn_for_params(
                G_raw=float(G_val),
                Wi_raw=int(Wi_val),
                result=result,
                require_in_box=require_in_box,
            )

            if len(df_roots) == 0:
                continue

            df_bad = df_roots[
                (df_roots["label"] == "C")
                & (df_roots["stable_sign"] == True)
                & (df_roots["Iprime"] > -1.2)
                & (df_roots["Iprime"] < 0.0)
                & (df_roots["Idoubleprime"].abs() < 0.2)
            ].copy()

            if len(df_bad) > 0:
                all_neg.append(df_bad)

    if len(all_neg) == 0:
        return pd.DataFrame()

    df_neg = pd.concat(all_neg, ignore_index=True)

    # remove near-duplicates
    df_neg = df_neg.drop_duplicates(
        subset=["G", "Wi", "Iprime", "Idoubleprime"]
    ).reset_index(drop=True)

    return df_neg


# ============================================================
# 7. Evaluation
# ============================================================
@torch.no_grad()
def evaluate_model_nn(df_eval, result, device="cpu"):
    model = result["model"]
    scaler = result["scaler"]
    h = result["hyper"]

    x_raw = df_eval["I'"].to_numpy(dtype=np.float64)
    y_raw = df_eval["I''"].to_numpy(dtype=np.float64)
    r1_raw = df_eval["G"].to_numpy(dtype=np.float64)
    r2_raw = df_eval["Wi"].to_numpy(dtype=np.float64)

    x_np, y_np, r1_np, r2_np = apply_param_scaler(
        x_raw, y_raw, r1_raw, r2_raw, scaler
    )

    x = torch.tensor(x_np, dtype=torch.float64, device=device)
    y = torch.tensor(y_np, dtype=torch.float64, device=device)
    r1 = torch.tensor(r1_np, dtype=torch.float64, device=device)
    r2 = torch.tensor(r2_np, dtype=torch.float64, device=device)

    # Need gradients, so temporarily leave no_grad
    with torch.enable_grad():
        f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_nn(
            model, x, y, r1, r2
        )

    Leq = (f1.pow(2) + f2.pow(2)).mean()

    trJ = df1_dx + df2_dy
    detJ = df1_dx * df2_dy - df1_dy * df2_dx

    alpha = h["alpha"]
    beta = h["beta"]

    return {
        "n": len(df_eval),
        "Leq": Leq.item(),
        "tr_ok": (trJ < -alpha).double().mean().item(),
        "det_ok": (detJ > beta).double().mean().item(),
        "joint_ok": ((trJ < -alpha) & (detJ > beta)).double().mean().item(),
        "trJ_mean": trJ.mean().item(),
        "trJ_min": trJ.min().item(),
        "trJ_max": trJ.max().item(),
        "detJ_mean": detJ.mean().item(),
        "detJ_min": detJ.min().item(),
        "detJ_max": detJ.max().item(),
    }


# ============================================================
# 8. Raw point evaluation
def evaluate_point_raw_nn(x_val, y_val, G_raw, Wi_raw, result):
    model = result["model"]
    scaler = result["scaler"]
    device = result["hyper"]["device"]

    x = torch.tensor([x_val], dtype=torch.float64, device=device)
    y = torch.tensor([y_val], dtype=torch.float64, device=device)
    r1 = torch.tensor([G_raw / scaler["sr1"]], dtype=torch.float64, device=device)
    r2 = torch.tensor([Wi_raw / scaler["sr2"]], dtype=torch.float64, device=device)

    f1, f2, df1_dx, df1_dy, df2_dx, df2_dy = compute_fJ_from_nn(
        model, x, y, r1, r2
    )

    f1_val = float(f1.detach().cpu().numpy()[0])
    f2_val = float(f2.detach().cpu().numpy()[0])

    df1_dx_val = float(df1_dx.detach().cpu().numpy()[0])
    df1_dy_val = float(df1_dy.detach().cpu().numpy()[0])
    df2_dx_val = float(df2_dx.detach().cpu().numpy()[0])
    df2_dy_val = float(df2_dy.detach().cpu().numpy()[0])

    J = np.array(
        [
            [df1_dx_val, df1_dy_val],
            [df2_dx_val, df2_dy_val],
        ]
    )

    trJ = float(np.trace(J))
    detJ = float(np.linalg.det(J))
    res_norm = float(np.sqrt(f1_val**2 + f2_val**2))

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
# 9. State classification
def classify_state_raw(x_raw, y_raw, margin_x=0.15, margin_y=0.15):
    if abs(y_raw) < margin_y:
        if abs(x_raw) < margin_x:
            return "symmetric"
        elif x_raw > margin_x:
            return "D"
        elif x_raw < -margin_x:
            return "C"
        else:
            return "other"
    if x_raw > margin_x and y_raw > margin_y:
        return "AD+"
    if x_raw > margin_x and y_raw < -margin_y:
        return "AD-"
    else:
        return "other"


# ============================================================
# 9.1 Mirror-pair correction (ported from
#     polynomial_symmetry_restricted_no_augmentation.py /
#     neural_network_symmetry_projected.py).
#     The averaging construction in DynamicalSystemNN.forward()
#     enforces f1(x,-y)=f1(x,y) and f2(x,-y)=-f2(x,y) exactly, so any
#     stable root (x,y) with y != 0 is guaranteed a stable mirror
#     partner (x,-y) with identical trace/det. The pointwise
#     threshold (margin_x, margin_y) can still mislabel a genuine
#     reflection pair as two "D" roots when |y| is just under the
#     threshold; this override fixes that without changing the
#     threshold itself.
# ============================================================
MIRROR_EPS_X = 1e-3
MIRROR_EPS_Y_SUM = 1e-3
MIRROR_EPS_SEP = 1e-3


def apply_mirror_pair_correction_nn(
    roots,
    eps_x=MIRROR_EPS_X,
    eps_y_sum=MIRROR_EPS_Y_SUM,
    eps_sep=MIRROR_EPS_SEP,
    margin_x=0.15,
):
    # NOTE: this must only be called on roots that have ALREADY been
    # filtered down to the accepted set (stable_sign, residual_tol,
    # and in_box if required) — find_roots_nn_for_params() only ever
    # appends accepted roots to roots_found, so no extra guard is
    # needed here (unlike the polynomial version, which used to run
    # this on the raw, unfiltered root table).
    #
    # This override only ever applies to a pair whose ORIGINAL
    # threshold label was "D" on both sides. A same-x mirror pair on
    # the C side (x < -margin_x, both labeled "C") satisfies the same
    # (x1≈x2, y1≈-y2) geometry but must NOT be promoted to AD+/AD- —
    # AD is defined with x > margin_x, so a C-branch mirror pair
    # staying "C" is the physically correct outcome. This matters
    # especially here: collect_negative_c_roots_from_pink_region()
    # specifically hunts for spurious stable "C" roots, and a
    # C-vs-C mirror pair in that region must not be able to escape
    # detection by getting relabeled AD+/AD- first.
    def _is_d_branch(r):
        return r["label"] == "D" and r["Iprime"] > margin_x

    roots = [r.copy() for r in roots]
    used = [False] * len(roots)

    for i in range(len(roots)):
        if used[i] or not _is_d_branch(roots[i]):
            continue
        xi = roots[i]["Iprime"]
        yi = roots[i]["Idoubleprime"]

        for j in range(i + 1, len(roots)):
            if used[j] or not _is_d_branch(roots[j]):
                continue
            xj = roots[j]["Iprime"]
            yj = roots[j]["Idoubleprime"]

            if (
                abs(xi - xj) < eps_x
                and abs(yi + yj) < eps_y_sum
                and yi * yj < 0
                and abs(yi - yj) > eps_sep
            ):
                if yi > yj:
                    roots[i]["label"] = "AD+"
                    roots[j]["label"] = "AD-"
                else:
                    roots[i]["label"] = "AD-"
                    roots[j]["label"] = "AD+"
                used[i] = True
                used[j] = True
                break

    return roots


# ============================================================
# 10. Root finding for NN
def find_roots_nn_for_params(
    G_raw,
    Wi_raw,
    result,
    x_bounds=(-1.1, 1.1),
    y_bounds=(-1.1, 1.1),
    n_grid=41,
    residual_tol=1e-6,
    duplicate_tol=1e-4,
    require_in_box=False,
):
    """Solve f1(x, y, G, Wi) = 0, f2(x, y, G, Wi) = 0 for equilibria at
    a fixed (G, Wi) via a multi-start search over an initial grid.
    """
    # n_grid x n_grid initial guesses sampled uniformly over the box
    xs = np.linspace(x_bounds[0], x_bounds[1], n_grid)
    ys = np.linspace(y_bounds[0], y_bounds[1], n_grid)

    roots_found = []

    def F_np(z):
        x_val, y_val = z
        out = evaluate_point_raw_nn(
            x_val=x_val,
            y_val=y_val,
            G_raw=G_raw,
            Wi_raw=Wi_raw,
            result=result,
        )
        return np.array([out["F1"], out["F2"]], dtype=float)

    for x0 in xs:
        for y0 in ys:
            sol = root(F_np, np.array([x0, y0]), method="hybr")

            if not sol.success:
                continue

            x_sol, y_sol = sol.x
            out = evaluate_point_raw_nn(
                x_val=x_sol,
                y_val=y_sol,
                G_raw=G_raw,
                Wi_raw=Wi_raw,
                result=result,
            )

            if out["res_norm"] > residual_tol:
                continue

            in_box = (
                x_bounds[0] <= x_sol <= x_bounds[1]
                and y_bounds[0] <= y_sol <= y_bounds[1]
            )

            if require_in_box and not in_box:
                continue

            duplicate = False
            for old in roots_found:
                dist = np.sqrt(
                    (old["Iprime"] - x_sol) ** 2
                    + (old["Idoubleprime"] - y_sol) ** 2
                )
                if dist < duplicate_tol:
                    duplicate = True
                    break

            # Many initial guesses converge to the same root; treat two
            # solutions as identical when their Euclidean distance in
            # the (x, y) plane is below duplicate_tol.
            if duplicate:
                continue

            out["G"] = G_raw
            out["Wi"] = Wi_raw
            out["in_box"] = in_box
            out["label"] = classify_state_raw(x_sol, y_sol)

            if out["stable_sign"]:
                roots_found.append(out)

    if len(roots_found) == 0:
        return pd.DataFrame(
            columns=[
                "G",
                "Wi",
                "Iprime",
                "Idoubleprime",
                "Itotal",
                "F1",
                "F2",
                "res_norm",
                "trJ",
                "detJ",
                "stable_sign",
                "in_box",
                "label",
            ]
        )

    roots_found = apply_mirror_pair_correction_nn(roots_found)

    return pd.DataFrame(roots_found)


def scan_wi_and_collect_roots_nn(
    G_fixed,
    Wi_start,
    Wi_end,
    Wi_step,
    result,
    require_in_box=False,
):
    """Scan a range of Wi values at fixed G, calling
    find_roots_nn_for_params once per Wi.
    """
    all_rows = []

    for Wi in range(Wi_start, Wi_end + 1, Wi_step):
        print(f"Solving NN roots for G={G_fixed}, Wi={Wi}")

        df_roots = find_roots_nn_for_params(
            G_raw=G_fixed,
            Wi_raw=Wi,
            result=result,
            require_in_box=require_in_box,
        )

        if len(df_roots) > 0:
            all_rows.append(df_roots)

    if len(all_rows) == 0:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ============================================================
# 11. Plot
# ============================================================
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
    Ported from polynomial_symmetry_restricted_no_augmentation.py so
    the NN and polynomial scripts share the same figure style.
    """
    # Marker and color are both fixed per state (not derived from
    # matplotlib's per-axes color cycle), and the legend is built
    # from a fixed proxy set covering every state seen anywhere in
    # the figure. This matches the phase-diagram convention and
    # Hopkins et al. Fig. 3's AD+/AD-/C symbols (red up-triangle,
    # blue down-triangle, orange square).
    style_map = {
        "symmetric": {"marker": "x", "color": "black"},
        # D uses tab:green here (not blue) to avoid ambiguity with AD-,
        # which the bifurcation diagram shows as a separate branch.
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
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows), sharex="col"
    )
    if n_cols == 1:
        axes = axes.reshape(3, 1)

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

        # Ground-truth overlay (experimental observations): black open circles.
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
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        eps_save_path = Path(save_path).with_suffix(".eps")
        fig.savefig(eps_save_path, format="eps", dpi=600, bbox_inches="tight")

    plt.show()


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
    ax.set_xlim(0.49, 0.635)
    ax.set_ylim(8, 300)
    # EPS doesn't support transparency: use a solid light-gray color
    # for the grid instead of alpha, and an opaque legend frame.
    ax.grid(True, which="both", linestyle="--", color="lightgray")
    ax.legend(loc="lower left", fontsize=11, framealpha=1.0)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        eps_save_path = Path(save_path).with_suffix(".eps")
        fig.savefig(eps_save_path, format="eps", dpi=600, bbox_inches="tight")

    plt.show()


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    # ========================================================
    # 0. User-controlled settings
    # ========================================================
    GLOBAL_SEED = 7
    SPLIT_SEED = 77

    DATA_PATH = "data/data_flow.xlsx"
    # Each script gets its own subfolder so outputs don't mix together.
    OUT_DIR = Path("paper_figures") / Path(__file__).stem

    TRAIN_FRAC = 0.8
    DEVICE = "cpu"

    # NN training settings
    HIDDEN_DIM = 50
    LAMBDA_STAB = 50
    ALPHA = 0.09
    BETA = 1e-3
    W_DET = 50.0
    LR = 5e-4
    WEIGHT_DECAY = 0.0
    STEPS = 60000
    LR_MILESTONES = (20000, 40000)
    LR_GAMMA = 0.5
    STANDARDIZE = True

    # Negative-root ("wrong stable C") iterative correction settings
    N_NEG_ITERS = 3
    LAMBDA_NEG = 1.0
    GAMMA_NEG = 1e-4
    NEG_FINE_TUNE_STEPS = 2000
    NEG_REGION_G_MIN = 0.50
    NEG_REGION_G_MAX = 0.55
    NEG_REGION_G_STEP = 0.01
    NEG_REGION_WI_MIN = 100
    NEG_REGION_WI_MAX = 200
    NEG_REGION_WI_STEP = 5

    # Bifurcation scan settings
    G_FIXED_LIST = [0.5, 0.588, 0.603]
    BIF_WI_START = 1
    BIF_WI_END = 200
    BIF_WI_STEP = 5
    # Unlike the sibling NN scripts, this one keeps the box filter on:
    # the negative-C fine-tuning makes the model's far-field behavior
    # less predictable than the baseline model, and without this
    # filter the root solver's out-of-box extrapolation artifacts
    # (e.g. spurious "stable C" roots at I' ~ -60, far outside the
    # physical data range of roughly [-1, 1.05]) leak into the
    # bifurcation and phase-diagram figures.
    REQUIRE_IN_BOX = True

    # Figure 4 phase diagram settings
    MAKE_FIGURE4 = True
    FIG4_G_GRID = np.round(np.arange(0.50, 0.631, 0.01), 3).tolist()
    FIG4_WI_GRID = list(range(8, 201, 5))

    # ========================================================
    # 1. Setup
    # ========================================================
    set_all_seeds(GLOBAL_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)

    # Use all data for training
    df_train = df.reset_index(drop=True)
    # Optional: keep df_test same as df only for evaluation/reporting
    df_test = df_train.copy()
    # No need for flip augmentation because symmetry is hard-coded
    # df_train = augment_data_flip_y(df_train, y_col="I''")

    print(f"Full data size : {len(df)}")
    print(f"Train size     : {len(df_train)}")
    print(f"Eval size      : {len(df_test)}")

    # ========================================================
    # 2. Train NN normal form
    # ========================================================
    result = fit_normal_form_nn(
        df_train=df_train,
        hidden_dim=HIDDEN_DIM,
        lambda_stab=LAMBDA_STAB,
        alpha=ALPHA,
        beta=BETA,
        w_det=W_DET,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        steps=STEPS,
        lr_milestones=LR_MILESTONES,
        lr_gamma=LR_GAMMA,
        standardize=STANDARDIZE,
        device=DEVICE,
        seed=GLOBAL_SEED,
    )

    # ========================================================
    # 2.1 Iterative negative-C correction: find spuriously stable
    # "C" roots in the pink (G,Wi) region and fine-tune them toward
    # instability, repeating until no more bad roots are found.
    # ========================================================
    df_neg_pool = pd.DataFrame()

    for neg_iter in range(N_NEG_ITERS):
        print(f"\n========== Negative correction iteration {neg_iter + 1} ==========")

        df_neg_new = collect_negative_c_roots_from_pink_region(
            result=result,
            G_min=NEG_REGION_G_MIN,
            G_max=NEG_REGION_G_MAX,
            G_step=NEG_REGION_G_STEP,
            Wi_min=NEG_REGION_WI_MIN,
            Wi_max=NEG_REGION_WI_MAX,
            Wi_step=NEG_REGION_WI_STEP,
            require_in_box=True,
        )

        print(f"New negative C roots this iter: {len(df_neg_new)}")

        if len(df_neg_new) > 0:
            df_neg_pool = pd.concat(
                [df_neg_pool, df_neg_new],
                ignore_index=True,
            )

            # remove exact duplicates
            df_neg_pool = df_neg_pool.drop_duplicates(
                subset=["G", "Wi", "Iprime", "Idoubleprime"]
            ).reset_index(drop=True)

        print(f"Total negative pool size: {len(df_neg_pool)}")

        if len(df_neg_pool) == 0:
            print("No wrong stable C roots found. Stop negative correction.")
            break

        neg_csv_path = OUT_DIR / f"negative_C_roots_pool_iter{neg_iter + 1}.csv"
        df_neg_pool.to_csv(neg_csv_path, index=False)
        print(f"Saved negative pool to: {neg_csv_path}")

        result = fine_tune_with_negative_roots_nn(
            result=result,
            df_train=df_train,
            df_neg=df_neg_pool,     # note: uses the full pool, not just df_neg_new
            lambda_stab=LAMBDA_STAB,
            lambda_neg=LAMBDA_NEG,
            gamma_neg=GAMMA_NEG,
            alpha=ALPHA,
            beta=BETA,
            w_det=W_DET,
            lr=1e-4,
            weight_decay=WEIGHT_DECAY,
            steps=NEG_FINE_TUNE_STEPS,
            device=DEVICE,
        )

    # ========================================================
    # 3. Evaluate train/test
    # ========================================================
    train_metrics = evaluate_model_nn(df_train, result, device=DEVICE)
    test_metrics = evaluate_model_nn(df_test, result, device=DEVICE)

    print("\n[Train metrics]")
    print(train_metrics)

    print("\n[Test metrics]")
    print(test_metrics)

    # ========================================================
    # 4. Fixed-G bifurcation scans — combined into one figure
    # ========================================================
    df_scan_by_G = {}
    for G_fixed in G_FIXED_LIST:
        df_scan = scan_wi_and_collect_roots_nn(
            G_fixed=G_fixed,
            Wi_start=BIF_WI_START,
            Wi_end=BIF_WI_END,
            Wi_step=BIF_WI_STEP,
            result=result,
            require_in_box=REQUIRE_IN_BOX,
        )

        print(f"\n[Scanned stable NN roots for G={G_fixed}]")
        print(df_scan.head())
        print(f"Total kept roots: {len(df_scan)}")

        csv_path = OUT_DIR / f"nn_roots_G{str(G_fixed).replace('.', 'p')}.csv"
        df_scan.to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")

        df_scan_by_G[G_fixed] = df_scan

    fig_path = OUT_DIR / "nn_bifurcation_combined.png"
    make_combined_bifurcation_plot(
        df_scan_by_G=df_scan_by_G,
        G_order=G_FIXED_LIST,
        df_gt=df,          # original non-augmented data
        save_path=fig_path,
    )
    print(f"Saved combined figure to: {fig_path}")

    # ========================================================
    # 5. Figure 4 style phase diagram
    # ========================================================
    if MAKE_FIGURE4:
        print("\n--- Generating Figure 4 style phase diagram ---")

        phase_roots_csv = OUT_DIR / "nn_figure4_phase_roots.csv"
        phase_summary_csv = OUT_DIR / "nn_figure4_phase_summary.csv"
        phase_fig_path = OUT_DIR / "nn_figure4_phase_diagram.png"

        # The (G, Wi) sweep is expensive. If a previous run already
        # saved results, load them instead of recomputing. Delete the
        # CSVs (or set FORCE_RECOMPUTE_PHASE=True) to force a fresh sweep.
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
            all_phase_rows = []

            fig4_wi_start = min(FIG4_WI_GRID)
            fig4_wi_end = max(FIG4_WI_GRID)
            fig4_wi_step = FIG4_WI_GRID[1] - FIG4_WI_GRID[0]

            for G_val in FIG4_G_GRID:
                df_g = scan_wi_and_collect_roots_nn(
                    G_fixed=G_val,
                    Wi_start=fig4_wi_start,
                    Wi_end=fig4_wi_end,
                    Wi_step=fig4_wi_step,
                    result=result,
                    require_in_box=REQUIRE_IN_BOX,
                )

                if len(df_g) > 0:
                    all_phase_rows.append(df_g)

            if len(all_phase_rows) == 0:
                df_all_grid = pd.DataFrame()
            else:
                df_all_grid = pd.concat(all_phase_rows, ignore_index=True)

            df_phase = generate_phase_diagram_data(
                df_all_roots=df_all_grid,
                G_grid=FIG4_G_GRID,
                Wi_grid=FIG4_WI_GRID,
            )

            df_all_grid.to_csv(phase_roots_csv, index=False)
            df_phase.to_csv(phase_summary_csv, index=False)

        print("\n[Phase counts]")
        print(df_phase["phase"].value_counts())

        plot_figure4_phase_diagram(
            df_phase=df_phase,
            save_path=phase_fig_path,
        )

        print(f"Saved roots to: {phase_roots_csv}")
        print(f"Saved phase summary to: {phase_summary_csv}")
        print(f"Saved phase figure to: {phase_fig_path}")
