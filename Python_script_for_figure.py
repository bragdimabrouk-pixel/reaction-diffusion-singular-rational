"""
Unified script for:
  "Exact Rational Solutions, Pole Dynamics, and a Spectral Bound for a
   Reaction-Diffusion Equation with Time-Dependent Diffusion and Linear Advection"

Combines:
  1. Independent verification of constants, identities, and eigenvalues.
  2. Figure generation:
       - fig_singularity_collapse.{pdf, png}  (Prop. 4.1, Cor. 4.2)
       - fig_effective_potential.{pdf, png}   (Lemma 6.3)

Requirements:
  pip install numpy scipy sympy mpmath matplotlib

Run:
  python combined_studio.py
"""

import numpy as np
import sympy as sp
from scipy.linalg import eigh_tridiagonal
from mpmath import mp, ellipfun, diff as mdiff, sqrt as msqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# Plot styling
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "text.usetex": False,
})


# ==============================================================================
# Part 1: Independent verification of constants and identities
# ==============================================================================
def verify_constants_and_identities():
    print("=" * 70)
    print("RUNNING VERIFICATION OF CONSTANTS AND IDENTITIES")
    print("=" * 70)

    t, x, eta, s = sp.symbols("t x eta s", real=True)
    C, A = sp.symbols("C A", positive=True)

    # 1. Theorem 3.1: residual of the PDE for the rational solution
    S = sp.sqrt(A + 48 * sp.exp(2 * t))
    u = C * sp.exp(t) / (x**2 * sp.exp(2 * t) - C / 8 * S)
    res = sp.diff(u, t) - ((C / S) * sp.diff(u, x, 2) + x * sp.diff(u, x) + u * (1 - u**2))
    for cv, av, tv, xv in [(1.7, 20, 0.3, 0.9), (2.3, 0, 0.7, 1.9), (1.1, -5, 1.4, 0.6)]:
        print("Thm 3.1 residual:", float(res.subs({C: cv, A: av, t: tv, x: xv})))

    # 2. Stationary profile W
    W = 4 * sp.sqrt(3) / (eta**2 - 6)
    print("W residual:", sp.simplify(sp.diff(W, eta, 2) + eta * sp.diff(W, eta) / 2 + W - W**3))

    # 3. Conjugation T L T^{-1} = -H
    psi = sp.Function("psi")(eta)
    Q = eta**2 / 16 - sp.Rational(3, 4) + 144 / (eta**2 - 6) ** 2
    phi = sp.exp(-eta**2 / 8) * psi
    Lphi = sp.diff(phi, eta, 2) + eta / 2 * sp.diff(phi, eta) + (1 - 3 * W**2) * phi
    print("Q identity:", sp.simplify(Lphi * sp.exp(eta**2 / 8) - (sp.diff(psi, eta, 2) - Q * psi)))

    # 4. Lemma 6.3
    c0 = sp.Rational(3, 4) * 3 ** sp.Rational(2, 3) - sp.Rational(3, 8)
    es = sp.sqrt(6 + 8 * 3 ** sp.Rational(2, 3))
    print("c0 =", sp.N(c0, 10), " Q(eta*) =", sp.N(Q.subs(eta, es), 10),
          " Q'(eta*) =", sp.N(sp.diff(Q, eta).subs(eta, es), 5), " eta* =", sp.N(es, 8),
          " Q(0) =", Q.subs(eta, 0))
    print("Laurent at sqrt(6):", sp.series(Q.subs(eta, sp.sqrt(6) + s), s, 0, 1))

    # 5. Prop. 4.1: asymptotic constant, log-derivative, value at the origin
    print("(3C^2/4)^(1/4) at C=4sqrt3:", sp.N((3 * (4 * sp.sqrt(3)) ** 2 / 4) ** sp.Rational(1, 4)))
    xp = sp.sqrt(C / 8) * S ** sp.Rational(1, 2) * sp.exp(-t)
    print("log-derivative identity:", sp.simplify(sp.diff(sp.log(xp), t) + (A + 24 * sp.exp(2 * t)) / (A + 48 * sp.exp(2 * t))))
    print("lim u(t,0) =", sp.limit((-C * sp.exp(t) / (C / 8 * S)).subs(A, 3), t, sp.oo), "= -2/sqrt(3) =", -2 / np.sqrt(3))

    # 6. Prop. 7.1: Jacobi sn solution
    mp.dps = 30
    for m in (0.2, 0.7, 0.95):
        v = lambda z: msqrt(2 * m / (1 + m)) * ellipfun("sn", z / msqrt(1 + m), m=m)
        print(f"Jacobi residual m={m}:", mdiff(v, 0.83, 2) + v(0.83) - v(0.83) ** 3)

    # 7. Numerical lowest eigenvalues of H = -d^2 + Q (finite differences).
    #    Dirichlet condition at the poles is consistent with the limit-point property
    #    (the L^2 solution behaves like s^3). NOT part of the proof; illustrates that
    #    c0 is a valid but non-sharp lower bound.
    Qf = lambda e: e**2 / 16 - 0.75 + 144 / (e**2 - 6) ** 2
    p = np.sqrt(6)
    def lowest(a, b, N, k=3):
        h = (b - a) / (N + 1)
        e = a + h * np.arange(1, N + 1)
        return eigh_tridiagonal(2 / h**2 + Qf(e), -np.ones(N - 1) / h**2,
                                select="i", select_range=(0, k - 1))[0]
    for N in (16000, 64000):
        print(f"I_+ = (sqrt6, 28), N={N}:", lowest(p, 28, N))
        print(f"I_0 = (-sqrt6, sqrt6), N={N}:", lowest(-p, p, N // 2))
    print("c0 =", float(c0))
    print()


# ==============================================================================
# Part 2: Figure Generation
# ==============================================================================
def generate_figures():
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Figure 1: pole curves x_pm(t) = +-sqrt(C/8) (A + 48 e^{2t})^{1/4} e^{-t}
    # --------------------------------------------------------------------------
    C = 4.0 * np.sqrt(3.0)          # canonical constant: gives f(t) = 6 e^t when A = 0
    t = np.linspace(-1.5, 4.0, 800)

    def pole(t, A, C):
        """Positive pole location x_+(t), eq. (4.1) of the paper."""
        return np.sqrt(C / 8.0) * (A + 48.0 * np.exp(2.0 * t)) ** 0.25 * np.exp(-t)

    x_can = pole(t, 0.0, C)
    # consistency checks against the closed forms in Prop. 4.1
    assert np.allclose(x_can, np.sqrt(6.0) * np.exp(-t / 2.0))
    assert np.isclose((3.0 * C**2 / 4.0) ** 0.25, np.sqrt(6.0))

    A = 20.0
    x_gen = pole(t, A, C)

    fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(t, x_can, color="#003366", label=r"Canonical branches $x_\pm(t)$ ($A=0$)")
    ax1.plot(t, -x_can, color="#003366")
    ax1.plot(t, x_gen, color="#990000", linestyle="--", linewidth=1.4,
             label=r"Branches for $A=20$")
    ax1.plot(t, -x_gen, color="#990000", linestyle="--", linewidth=1.4)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1)

    ax1.set_xlabel(r"Time $t$")
    ax1.set_ylabel(r"Pole location $x$")
    ax1.set_title(r"Collapse of the pole set: $|x_\pm(t)|\sim (3C^2/4)^{1/4}e^{-t/2}$")
    ax1.set_xlim([-1.5, 4.0])
    ax1.set_ylim([-5.0, 5.0])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)
    fig1.tight_layout()
    fig1.savefig("fig_singularity_collapse.pdf")
    fig1.savefig("fig_singularity_collapse.png", dpi=300)
    print("Saved: fig_singularity_collapse.pdf / png")
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: potential Q(eta) and its lower bound c0 (Lemma 6.3)
    # --------------------------------------------------------------------------
    def Q(eta):
        return eta**2 / 16.0 - 0.75 + 144.0 / (eta**2 - 6.0) ** 2

    pole_eta = np.sqrt(6.0)
    c0 = 0.75 * 3.0 ** (2.0 / 3.0) - 0.375                 # = 1.185062...
    eta_star = np.sqrt(6.0 + 8.0 * 3.0 ** (2.0 / 3.0))     # = 4.7582...
    Q0 = Q(0.0)                                            # = 13/4

    # consistency checks against Lemma 6.3
    assert np.isclose(Q(eta_star), c0)
    assert np.isclose(Q0, 13.0 / 4.0)
    grid = np.concatenate([np.linspace(-30, -pole_eta - 1e-3, 400000),
                           np.linspace(-pole_eta + 1e-3, pole_eta - 1e-3, 400000),
                           np.linspace(pole_eta + 1e-3, 30, 400000)])
    assert Q(grid).min() >= c0 - 1e-9

    off = 0.15
    eta_left = np.linspace(-7.0, -pole_eta - off, 400)
    eta_mid = np.linspace(-pole_eta + off, pole_eta - off, 400)
    eta_right = np.linspace(pole_eta + off, 7.0, 400)

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.4))
    for seg, lab in ((eta_left, r"Potential $Q(\eta)$"), (eta_mid, None), (eta_right, None)):
        ax2.plot(seg, Q(seg), color="#002244", label=lab)

    ax2.axvline(-pole_eta, color="black", linestyle=":", linewidth=1.2,
                label=r"Inverse-square singularities $\eta=\pm\sqrt{6}$")
    ax2.axvline(pole_eta, color="black", linestyle=":", linewidth=1.2)
    ax2.axhline(c0, color="#cc0000", linestyle="--", linewidth=1.3,
                label=rf"Lower bound $c_0\approx{c0:.3f}$")

    ax2.scatter([0], [Q0], color="#0066cc", zorder=5, s=35)
    ax2.scatter([eta_star, -eta_star], [c0, c0], color="#cc0000", zorder=5, s=35)
    ax2.annotate(rf"$Q(0)={Q0:.2f}$", xy=(0, Q0), xytext=(0.4, 4.5),
                 arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax2.annotate(rf"$c_0\approx{c0:.3f}$", xy=(eta_star, c0), xytext=(eta_star - 1.2, c0 + 2.5),
                 arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)

    ax2.set_xlabel(r"Similarity coordinate $\eta$")
    ax2.set_ylabel(r"Effective potential $Q(\eta)$")
    ax2.set_title(r"Effective potential $Q(\eta)$ and the lower bound $Q\geq c_0>0$")
    ax2.set_xlim([-7.0, 7.0])
    ax2.set_ylim([0.0, 14.0])
    ax2.grid(True, linestyle=":", alpha=0.6)
    # legend placed below the axes so that it does not cover the curve
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=True, fontsize=9)
    fig2.tight_layout()
    fig2.savefig("fig_effective_potential.pdf", bbox_inches="tight")
    fig2.savefig("fig_effective_potential.png", dpi=300, bbox_inches="tight")
    print("Saved: fig_effective_potential.pdf / png")
    plt.close(fig2)


if __name__ == "__main__":
    verify_constants_and_identities()
    generate_figures()