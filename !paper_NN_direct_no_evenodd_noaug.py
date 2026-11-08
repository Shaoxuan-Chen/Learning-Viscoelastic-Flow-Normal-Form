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
# 4. Neural network model
# ============================================================
class DynamicalSystemNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.shared_trunk = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.head_f1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.head_f2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, y, r1, r2):
        inputs = torch.stack([x, y, r1, r2], dim=1)
        h = self.shared_trunk(inputs)

        f1 = self.head_f1(h).squeeze(-1)
        f2 = self.head_f2(h).squeeze(-1)

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
       #e.g. I''>0.15(像AD), I' < 0.15（不满足AD）---bifurcation transition region
       #e.g. I'= -0.03(like symmetric), I'' = -0.18(not symmetric)
       #not satisfy phsics, e.g. I'<0, I''>0


# ============================================================
# 10. Root finding for NN
def find_roots_nn_for_params(
    G_raw,
    Wi_raw,
    result,
    x_bounds=(-1.1, 1.1),
    y_bounds=(-1.1, 1.1),
    # [TEST] Temporarily set back to 1.1 (matching x_bounds/y_bounds)
    # to compare against the 1.5 version — paper Sec 2.1 says this
    # should be 1.5, kept as a separate parameter either way.
    classify_bounds=(-1.1, 1.1),
    n_grid=41,
    residual_tol=1e-6,
    duplicate_tol=1e-4,
    require_in_box=False,
):
    """固定一个参数： G, W
然后要解：
f_1(x,y,G,Wi)=0
f_2(x,y,G,Wi)=0
---找 equilibrium / steady state
    """
    xs = np.linspace(x_bounds[0], x_bounds[1], n_grid)
    ys = np.linspace(y_bounds[0], y_bounds[1], n_grid)
      #10.1 xs = np.linspace(-1.1, 1.1, 41)
      #     ys = np.linspace(-1.1, 1.1, 41) initial guss of the初始点---total 1681 inital guess

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
     # 10.2 get the [f1, f2]

    for x0 in xs:
        for y0 in ys:
            sol = root(F_np, np.array([x0, y0]), method="hybr")
            #10.3 对初始点解： if  res\_norm=\sqrt{f_1^2+f_2^2} > margin: continue

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
                classify_bounds[0] <= x_sol <= classify_bounds[1]
                and classify_bounds[0] <= y_sol <= classify_bounds[1]
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

            if duplicate:
                continue
            #很多 initial guesses 会收敛到同一个 root，所以这里检查新 root 和已有 root 的距离： \sqrt{(x_{\text{old}}-x_{\text{new}})^2+(y_{\text{old}}-y_{\text{new}})^2}
                    #如果小于：duplicate_tol=1e-4, 就认为是同一个 root，不重复保存。

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

    return pd.DataFrame(roots_found)


def scan_wi_and_collect_roots_nn(
    G_fixed,
    Wi_start,
    Wi_end,
    Wi_step,
    result,
    require_in_box=False,
    n_grid=41,
):
    """
    固定一个 G，扫很多 Wi--- 实际扫的是：Wi=1,6,11,...,196. 每个 Wi 都调用一次：
    """
    all_rows = []

    for Wi in range(Wi_start, Wi_end + 1, Wi_step):
        print(f"Solving NN roots for G={G_fixed}, Wi={Wi}")

        df_roots = find_roots_nn_for_params(
            G_raw=G_fixed,
            Wi_raw=Wi,
            result=result,
            require_in_box=require_in_box,
            n_grid=n_grid,
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
    Ported from paper_polynomial_evenodd_noaug.py so the NN and
    polynomial scripts share the same figure style.
    """
    # [FIX] Marker AND color are both fixed per state here (not just
    # marker). Previously only marker was set, so matplotlib assigned
    # colors by plot order within each column's independent color
    # cycle — since labels_present differs per G, the same color could
    # mean a different state in different columns.
    # [FIX] Unified with the phase-diagram color/marker convention and
    # with Hopkins et al. Fig. 3's AD+/AD-/C symbols (red up-triangle,
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

    # [FIX] Track every label seen across ALL columns so the legend can
    # be built once, independent of which column happens to contain it.
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

    # [FIX] Build the legend from a fixed set of proxy handles covering
    # every state seen anywhere in the figure, in a stable order — not
    # from whatever happened to be plotted in the last column.
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
    # [FIX] EPS doesn't support transparency: use a solid light-gray
    # color for the grid instead of alpha, and an opaque legend frame.
    ax.grid(True, which="both", linestyle="--", color="lightgray")
    ax.legend(loc="lower left", fontsize=11, framealpha=1.0)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        eps_save_path = Path(save_path).with_suffix(".eps")
        fig.savefig(eps_save_path, format="eps", dpi=600, bbox_inches="tight")

    plt.show()








##################################################################
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
    HIDDEN_DIM = 128
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

    # [PREVIEW] The real bottleneck is n_grid (multi-start root search
    # density): the original uses 41 (1681 scipy.optimize.root calls
    # PER (G,Wi) point, each through the NN's autograd Jacobian). This
    # is what was actually making everything slow, not the outer grid.
    # Cut it down hard for a quick look; raise it back to 41 once
    # you're ready for a real run.
    PREVIEW_N_GRID = 9   # 9x9 = 81 starts/point, vs 1681 originally

    # Bifurcation scan settings
    # [MATCH] Wi grid matched to !paper_polynomial_evenodd_noaug.py
    # (Wi_step=5 -> Wi=1,6,...,196). n_grid (root-search density)
    # above is left untouched.
    G_FIXED_LIST = [0.5, 0.588, 0.603]
    BIF_WI_START = 1
    BIF_WI_END = 200
    BIF_WI_STEP = 5
    REQUIRE_IN_BOX = False

    # Figure 4 phase diagram settings
    # [MATCH] Grid matched to !paper_polynomial_evenodd_noaug.py's
    # paper-exact grid (G_STEP=0.005, WI_STEP=5).
    MAKE_FIGURE4 = True
    FIG4_G_GRID = np.round(np.arange(0.50, 0.63 + 1e-9, 0.01), 3).tolist()
    FIG4_WI_GRID = list(range(8, 198 + 1, 5))


    # ========================================================
    # 1. Setup
    # ========================================================
    set_all_seeds(GLOBAL_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)

        # Use full original data
    df_train = df.reset_index(drop=True)

    # [FIX] Paper Sec 2.4.1: "The direct neural-network model is
    # trained using the original equilibrium observations without
    # reflected-data augmentation." Augmentation was tested and found
    # to make the direct NN reconstruction worse, so it's left off.
    # df_train = augment_data_flip_y(df_train, y_col="I''")

    # Use full original non-augmented data for evaluation/reporting
    df_test = df.reset_index(drop=True)

    print(f"Full original data size : {len(df)}")
    print(f"Train size (no augmentation) : {len(df_train)}")
    print(f"Eval size (same, no held-out split): {len(df_test)}")
        
    

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
            n_grid=PREVIEW_N_GRID,
        )

        print(f"\n[Scanned stable NN roots for G={G_fixed}]")
        print(df_scan.to_string())
        print(f"Total kept roots: {len(df_scan)}")

        df_scan_by_G[G_fixed] = df_scan

    fig_path = OUT_DIR / "nn_bifurcation_combined_preview.png"
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

        phase_roots_csv = OUT_DIR / "nn_figure4_phase_roots_preview.csv"
        phase_summary_csv = OUT_DIR / "nn_figure4_phase_summary_preview.csv"
        phase_fig_path = OUT_DIR / "nn_figure4_phase_diagram_preview.png"

        # [CACHE] The (G,Wi) sweep is expensive. If a previous run
        # already saved results, load them instead of recomputing.
        # Delete the CSVs (or set FORCE_RECOMPUTE_PHASE=True) to force
        # a fresh sweep.
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
                    n_grid=PREVIEW_N_GRID,
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
        print("\n[Phase grid]")
        print(df_phase.to_string())

        plot_figure4_phase_diagram(
            df_phase=df_phase,
            save_path=phase_fig_path,
        )

        print(f"Saved roots to: {phase_roots_csv}")
        print(f"Saved phase summary to: {phase_summary_csv}")
        print(f"Saved phase figure to: {phase_fig_path}")