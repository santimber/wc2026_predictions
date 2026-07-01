"""Diagnostics: is model confidence informative, and which weight scheme is robust?
Reuses calibrate.py's data + helpers (no API)."""
import numpy as np, pandas as pd
import calibrate as cb, predict_ensemble as pe

feats = pe.load_feats()
_, fixtures, actuals = pe.assemble()
A = pd.read_pickle(cb.A_CACHE)
Ag_df = A[A["outcome"].notna()]
gkeys = list(zip(Ag_df["home_team"], Ag_df["away_team"]))
y = Ag_df["outcome"].values
Ag = Ag_df[["p_home_win", "p_draw", "p_away_win"]].values
Bg = cb.loo_poisson(gkeys)
grp = feats[feats.tournament.eq("FIFA World Cup") & feats.outcome.notna() & feats.date.dt.year.eq(2026)]
dmax = min([0.16,0.2,0.24,0.28,0.32], key=lambda d: cb.ll(y, cb.elo_mat(grp, gkeys, d)))
Cg = cb.elo_mat(grp, gkeys, dmax)

# reliability: bin by max prob, report accuracy per bin
print("=== Reliability (does confidence track accuracy?) ===")
for nm, M in [("TabPFN", Ag), ("Poisson", Bg), ("Elo", Cg)]:
    conf = M.max(1); pred = M.argmax(1)
    correct = np.array([cb.CLASSES[p] for p in pred]) == y
    q = np.quantile(conf, [0, .5, 1.0])
    line = []
    for lo, hi, lab in [(q[0], q[1], "low "), (q[1], q[2]+1e-9, "high")]:
        msk = (conf >= lo) & (conf <= hi)
        if msk.sum(): line.append(f"{lab} conf n={msk.sum():2d} acc={correct[msk].mean():.0%}")
    print(f"  {nm:8s} meanconf={conf.mean():.2f}  " + " | ".join(line))

# weight schemes on temperature-calibrated models
Ts = {}
rng = np.random.RandomState(0); folds = rng.permutation(len(y)) % 5
for k, M in [("A", Ag), ("B", Bg), ("C", Cg)]:
    Ts[k] = min(np.arange(0.6, 8.01, 0.2),
                key=lambda T: np.mean([cb.ll(y[folds==f], cb.temp(M[folds==f], T)) for f in range(5)]))
AgT, BgT, CgT = cb.temp(Ag, Ts["A"]), cb.temp(Bg, Ts["B"]), cb.temp(Cg, Ts["C"])
mats = [AgT, BgT, CgT]
print(f"\nTemps A={Ts['A']:.1f} B={Ts['B']:.1f} C={Ts['C']:.1f}")

def cvll(w): return np.mean([cb.ll(y[folds==f], cb.blend([M[folds==f] for M in mats], w)) for f in range(5)])
schemes = {"TabPFN-only":(1,0,0), "Poisson-only":(0,1,0), "Elo-only":(0,0,1),
           "equal":(1/3,1/3,1/3), "A-heavy":(0.5,0.2,0.3)}
print("\n=== Weight schemes (calibrated) ===")
for nm, w in schemes.items():
    P = cb.blend(mats, w)
    print(f"  {nm:13s} w={tuple(round(x,2) for x in w)}  CVll={cvll(w):.3f}  acc={cb.acc(y,P):.0%}")

# R32 winner agreement across schemes
r32 = feats[feats.home_score.isna() & feats.tournament.eq("FIFA World Cup")]
rk = [(h,a) for h,a,_ in fixtures]
Ar = cb.temp(np.array([A.set_index(["home_team","away_team"]).loc[k,["p_home_win","p_draw","p_away_win"]].values for k in rk],dtype=float), Ts["A"])
Br = cb.temp(pe.model_b(fixtures), Ts["B"]); Cr = cb.temp(cb.elo_mat(r32, rk, dmax), Ts["C"])
rmats=[Ar,Br,Cr]
print("\n=== R32 winners by scheme (home/away pick) ===")
def winners(w):
    P=cb.blend(rmats,w); adv=P[:,0]+0.5*P[:,1]
    return [rk[i][0] if adv[i]>=.5 else rk[i][1] for i in range(len(rk))]
wsets={nm:winners(w) for nm,w in schemes.items()}
for i,(h,a) in enumerate(rk):
    picks={nm:wsets[nm][i] for nm in schemes}
    disagree = len(set(picks.values()))>1
    print(f"  {h[:14]:14s} v {a[:14]:14s}  " + " ".join(f"{nm[:4]}:{picks[nm][:10]}" for nm in schemes) + ("  <DIFF" if disagree else ""))
