# Learning Viscoelastic Flow Normal Form

Code for reproducing the polynomial and neural-network normal-form models
of viscoelastic flow past two microcylinders, fit to the experimental
data of Hopkins et al. (PRL 2021). Each script trains one model variant
from the paper, solves for its equilibria, and reproduces the
corresponding bifurcation diagram / phase diagram figures.

## Scripts

| Script | Paper section | Model |
|---|---|---|
| `!paper_polynomial_unrestricted_noaug.py` | Appendix B (ablation) | Fully unrestricted polynomial normal form, no reflection-symmetry constraint, no data augmentation. Reports only the bifurcation diagram (no phase diagram), matching the paper's methodology for this ablation. |
| `!paper_polynomial_evenodd_noaug.py` | Appendix B (ablation) | Polynomial normal form restricted to even-in-`I''` terms in `f1` / odd-in-`I''` terms in `f2`, no data augmentation. Reports bifurcation diagram and phase diagram, with a mirror-pair classification correction (see below). |
| `!paper_NN_direct_no_evenodd_noaug.py` | Sec. 2.4.1 / Appendix C | Direct neural-network vector field (shared trunk + two heads), no symmetry constraint, no augmentation. |
| `!paper_NN_avg_evenodd_noaug.py` | Sec. 2.4.2 | Symmetry-projected neural-network vector field: the network output is reflection-averaged so that `f1` is exact even and `f2` is exact odd in `I''`. |
| `!paper_NN_avg_evenodd_noaug_neg_C.py` | Sec. 2.4.2 + iterative refinement | Same symmetry-projected network as above, plus an additional iterative fine-tuning stage that penalizes spurious stable "C" equilibria found in a specific `(G, Wi)` sub-region. |

The fully-restricted polynomial model with data augmentation (Sec. 2.3,
the paper's main polynomial result) is intentionally not included here.

## Data

All scripts read `data/data_flow.xlsx` (columns `G`, `Wi`, `I'`, `I''`),
the experimental equilibrium observations the models are fit to.

## Running

```bash
python3 "!paper_NN_avg_evenodd_noaug.py"
```

Each script creates its own output subfolder under `paper_figures/<script_name>/`
containing the trained model's root CSVs and the bifurcation-diagram /
phase-diagram PNG + EPS figures.

The NN scripts (`!paper_NN_*`) are self-contained (training + equilibrium
search via `scipy.optimize.root`). The polynomial scripts
(`!paper_polynomial_*`) additionally solve for equilibria via
[Bertini](https://bertini.nd.edu/) (homotopy continuation), and hardcode
a local path to a Bertini installation:

```python
bertini_exec = "/Users/shaoxuanchen/Desktop/BertiniApple_v1.7/bertini"
bertini_workdir = Path("/Users/shaoxuanchen/Desktop/BertiniApple_v1.7/examples/viscoelastic_flow")
```

To run these two scripts elsewhere, install Bertini separately and edit
these two paths accordingly.
