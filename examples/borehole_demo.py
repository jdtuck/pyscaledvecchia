"""
Borehole function demo, d = 8 (only ~3 inputs really matter).

This is the illustrative computer-model example used in Sec. 4.3 of the
paper. Run with:

    python examples/borehole_demo.py
"""

import time

import numpy as np

from scaled_vecchia import ScaledVecchiaGP


def borehole(X):
    """Borehole function on [0,1]^8 (Morris et al. 1993)."""
    lo = np.array([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0])
    hi = np.array([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0])
    Z = lo + X * (hi - lo)
    rw, r, Tu, Hu, Tl, Hl, L, Kw = Z.T
    num = 2 * np.pi * Tu * (Hu - Hl)
    lr = np.log(r / rw)
    den = lr * (1 + 2 * L * Tu / (lr * rw ** 2 * Kw) + Tu / Tl)
    return num / den


def main():
    print("=" * 68)
    print("DEMO: borehole function, d = 8 (only ~3 inputs really matter)")
    print("=" * 68)
    rng = np.random.default_rng(0)
    n, ns, d = 4000, 1000, 8
    Xtr, Xte = rng.random((n, d)), rng.random((ns, d))
    ytr, yte = borehole(Xtr), borehole(Xte)

    t0 = time.time()
    gp = ScaledVecchiaGP(m_est=30, m_pred=100, nu=2.5, trend="constant",
                          nugget=1e-8, n_est=2000, verbose=True).fit(Xtr, ytr)
    t_fit = time.time() - t0

    t0 = time.time()
    mu, sd = gp.predict(Xte, return_std=True)
    t_pred = time.time() - t0

    rmse = float(np.sqrt(np.mean((mu - yte) ** 2)))
    triv = float(np.sqrt(np.mean((ytr.mean() - yte) ** 2)))
    cov95 = float(np.mean(np.abs(mu - yte) <= 1.96 * sd))
    print()
    print(gp.summary())
    print()
    print(f"  fit time            : {t_fit:.1f} s   ({n} runs)")
    print(f"  prediction time     : {t_pred:.1f} s   ({ns} test inputs)")
    print(f"  RMSE                : {rmse:.4g}")
    print(f"  RMSE (mean predictor): {triv:.4g}")
    print(f"  95% interval coverage: {cov95:.3f}")

    # joint prediction along a path through input space
    print()
    print("  --- joint prediction along a path through input space ---")
    t = np.linspace(0, 1, 200)[:, None]
    path = 0.5 + 0.4 * np.column_stack([np.sin(2 * np.pi * t * (l + 1) / d)
                                        for l in range(d)])
    res = gp.predict_joint(path, m=60, n_sim=50)
    truth = borehole(path)
    print(f"  joint path RMSE     : "
          f"{np.sqrt(np.mean((res['mean'] - truth) ** 2)):.4g}")
    print(f"  drew {res['samples'].shape[0]} joint sample paths of length "
          f"{res['samples'].shape[1]}")


if __name__ == "__main__":
    main()
