"""Reproduces Section 2 of the report: the defects of the 2024 notebook
(energy injection by the friction sign error, and finite-difference gradients that are solver noise)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
m, g = 1.0, 9.81
t_span = (0, 20); t_eval = np.linspace(*t_span, 1000)
c, k, mu, al = 0.5, 10.0, 0.1, 0.5   # original initial guess

def orig(t, y):
    x, v = y
    F = mu*m*g*np.exp(-al*abs(v))
    return [v, (-c*v - k*x + F*np.sign(v))/m]
def fixed_nobelt(t, y):   # same law, friction opposing velocity (no belt)
    x, v = y
    F = mu*m*g*np.exp(-al*abs(v))
    return [v, (-c*v - k*x - F*np.sign(v))/m]

so = solve_ivp(orig, t_span, [0, 1], t_eval=t_eval)
sf = solve_ivp(fixed_nobelt, t_span, [0, 1], t_eval=t_eval, max_step=1e-3)
E = lambda s: 0.5*m*s.y[1]**2 + 0.5*k*s.y[0]**2
# power delivered by the original friction term
P = mu*m*g*np.exp(-al*np.abs(so.y[1]))*np.abs(so.y[1])
print("orig: E0=%.3f  E(end)=%.3f  min/max E in last 5 s: %.3f %.3f"%(E(so)[0],E(so)[-1],E(so)[t_eval>15].min(),E(so)[t_eval>15].max()))
print("friction power always >= 0:", (P>=0).all(), " mean friction power last 5s %.3f W"%P[t_eval>15].mean())

# original objective (t_span 0..20 as in the SGD cell) and its finite-difference gradient
def objective(params, rtol=1e-3):
    c, k, mu, al = params
    def f(t, y):
        x, v = y
        return [v, (-c*v - k*x + mu*m*g*np.exp(-al*abs(v))*np.sign(v))/m]
    s = solve_ivp(f, t_span, [0, 1], t_eval=t_eval, rtol=rtol)
    e = 0.5*m*s.y[1]**2 + 0.5*c*0 + 0.5*k*s.y[0]**2
    return e.max() - e.min()
p0 = np.array([0.5, 10.0, 0.1, 0.5])
J0 = objective(p0)
fd = []
for j in range(4):
    pe = p0.copy(); pe[j] += 1e-5
    fd.append((objective(pe) - J0)/1e-5)
print("objective at start %.6f, FD gradient (h=1e-5):"%J0, np.round(fd, 3))
# reference gradient with tight tolerance and central differences, h=1e-3
gref = []
for j in range(4):
    a = p0.copy(); b = p0.copy(); a[j] += 1e-3; b[j] -= 1e-3
    gref.append((objective(a, 1e-10) - objective(b, 1e-10))/2e-3)
print("reference gradient (rtol 1e-10, central h=1e-3):", np.round(gref, 4))
# sweep of c at fine resolution to show the jagged objective
cs = np.linspace(0.5, 0.5005, 201)
Jc = [objective(np.array([cc, 10, 0.1, 0.5])) for cc in cs]
Jt = [objective(np.array([cc, 10, 0.1, 0.5]), 1e-10) for cc in cs]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
ax[0].plot(t_eval, E(so), "C3", label="original: +F·sign(ẋ)")
ax[0].plot(t_eval, E(sf), "C0", label="corrected sign: −F·sign(ẋ)")
ax[0].set(xlabel="t (s)", ylabel="mechanical energy (J)", title="(a) Original friction term adds energy")
ax[0].legend(fontsize=8)
ax[1].plot(so.y[0], so.y[1], "C3", lw=0.8, label="original")
ax[1].plot(sf.y[0], sf.y[1], "C0", lw=1, label="corrected sign")
ax[1].set(xlabel="x (m)", ylabel="ẋ (m/s)", title="(b) Self-sustained cycle with no energy source")
ax[1].legend(fontsize=8)
ax[2].plot((cs-0.5)*1e4, Jc, "C3", lw=1, label="default solver tolerance (original)")
ax[2].plot((cs-0.5)*1e4, Jt, "k", lw=2, label="tight tolerance (rtol 1e-10)")
ax[2].set(xlabel="c − 0.5  (×10⁻⁴ N·s/m)", ylabel="objective  max(E) − min(E)  (J)",
          title="(c) Objective is solver noise at h = 10⁻⁵", ylim=(0.52, 0.735))
ax[2].legend(fontsize=8, loc="lower center", framealpha=1)
plt.tight_layout(); plt.savefig("fig0_original_flaws.png", dpi=130)
print("saved fig0_original_flaws.png")
