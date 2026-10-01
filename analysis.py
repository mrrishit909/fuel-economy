"""Step 2: clean EPA test-car data and separate what buyers chose (weight, power) from engine technology.

    ./venv/bin/python analysis.py

Outcome: fuel consumption in gallons per 100 miles on EPA's lab tests (0.55 city + 0.45 highway), logged, so coefficients
are elasticities. Year effects measure how much less fuel a car of the SAME weight, power, body and drive used each year.
Standard errors are clustered by manufacturer (one maker's cars share engineering), and cross-validation holds out whole
models, because the same model is tested many times.
results/cleaning.json, year_effects.csv, coefficients.csv, decomposition.json, cv.json, robustness.csv, physics.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = Path(__file__).parent
FIRST, LAST = 2013, 2025          # 2012 file: only to recognise carried-over tests; 2026 file: a partial January release
HYBRID_NAME = (r"HYBRID|\bHEV\b|PHEV|PRIUS|\b\w+ ?\d{3}H\b|IONIQ|NIRO|INSIGHT|PRIME|FHEV|4XE|RECHARGE|E-TRON|\bT8\b|PLUG"
               r"|LAFERRARI|\bP1\b|REVUELTO|SF90|ARTURA|\b918\b|\bI8\b")   # strong and plug-in hybrids; 48-volt "MHEV" stay in
TRANS = {"Manual": "manual", "Automatic": "automatic", "Semi-Automatic": "automatic", "Continuously Variable": "CVT",
         "Selectable Continuously Variable (e.g. CVT with paddles)": "CVT", "Automated Manual": "automated manual",
         "Automated Manual- Selectable (e.g. Automated Manual with paddles)": "automated manual"}
DRIVE = {"2-Wheel Drive, Front": "front", "2-Wheel Drive, Rear": "rear", "All Wheel Drive": "all/4-wheel",
         "4-Wheel Drive": "all/4-wheel", "Part-time 4-Wheel Drive": "all/4-wheel"}
GROUP = {"land": "jaguar", "audi": "volkswagen"}   # same company filing under another name in some years
ATTRS = "log_weight + log_hp + C(vtype, Treatment('Car')) + C(drive, Treatment('front'))"
TECH = "log_displ + C(trans, Treatment('automatic')) + gears + cylinders"


def load():
    frames = []
    for p in sorted((HERE / "data" / "raw").iterdir()):
        d = pd.read_csv(p, encoding="latin1", low_memory=False) if p.suffix == ".csv" else pd.read_excel(p)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d.assign(year=int(p.stem)))
    return pd.concat(frames, ignore_index=True)


def clean(a, last=LAST):
    """Documented rules, counted. Test-level rules first, then one row per test vehicle."""
    steps = [("test results, model years 2012-2026", len(a))]
    rules = [
        ("gasoline (EPA certification fuel)", a["Test Fuel Type Description"] == "Tier 2 Cert Gasoline"),
        ("city (FTP) or highway test", a["Test Category"].isin(["FTP", "HWY"])),
        ("measured, not analytically derived", a["Analytically Derived FE?"] == "No"),
        ("not a police or emergency vehicle", a["Police - Emergency Vehicle?"] != "Y"),
    ]
    keep = pd.Series(True, index=a.index)
    for name, cond in rules:
        keep &= cond.fillna(False)
        steps.append((name, int(keep.sum())))
    t = a[keep].sort_values("year").drop_duplicates("Test Number")      # a carried-over test counts once, in its first year
    steps.append(("each test once, in its first model year (carry-overs dropped)", len(t)))
    t = t[t["year"].between(FIRST, last)]
    steps.append((f"first tested {FIRST}-{last} (2026 is a partial early release)", len(t)))

    t = t.assign(gpm=100 / t["RND_ADJ_FE"])
    key = ["year", "Test Vehicle ID", "Test Veh Configuration #"]
    cons = t.pivot_table(index=key, columns="Test Category", values="gpm", aggfunc="mean").dropna()   # average repeats in gal/100mi
    attrs = t.groupby(key).agg(make=("Represented Test Veh Make", "first"), model=("Represented Test Veh Model", "first"),
                               mfr=("Vehicle Manufacturer Name", "first"), vtype=("Vehicle Type", "first"),
                               weight=("Equivalent Test Weight (lbs.)", "first"), hp=("Rated Horsepower", "first"),
                               displ=("Test Veh Displacement (L)", "first"), cylinders=("# of Cylinders and Rotors", "first"),
                               gears=("# of Gears", "first"), trans=("Tested Transmission Type", "first"),
                               drive=("Drive System Description", "first"), coef_a=("Target Coef A (lbf)", "first"),
                               coef_c=("Target Coef C (lbf/mph**2)", "first"))
    v = cons.join(attrs).reset_index()
    steps.append(("vehicles with both a city and a highway result", len(v)))

    plug_in = set(a.loc[a["Test Category"] == "CD", "Test Vehicle ID"])           # charge-depleting tests = plug-in hybrid
    v["ratio"] = v["FTP"] / v["HWY"]                                              # conventional cars: highway ~1.5x better
    v["hybrid"] = (v["model"].astype(str).str.upper().str.contains(HYBRID_NAME) | v["Test Vehicle ID"].isin(plug_in)
                   | (v["ratio"] < 1.2))   # ponytail: name + ratio heuristic; 48-volt mild hybrids stay in, unflagged
    v["gpm"] = 0.55 * v["FTP"] + 0.45 * v["HWY"]
    v["trans"] = v["trans"].map(TRANS)
    v["drive"] = v["drive"].map(DRIVE)
    v["group"] = v["mfr"].str.lower().str.split().str[0].replace(GROUP)
    v["model_key"] = (v["make"].str.upper() + " " + v["model"].astype(str).str.upper().str.split().str[0])
    v["sample"] = ~v["hybrid"]
    steps.append(("not a hybrid or plug-in", int(v["sample"].sum())))
    v["usable"] = (v["hp"].between(50, 2000) & (v["hp"] != 999) & v["weight"].between(1500, 9000) & v["displ"].gt(0) & v["trans"].notna()
                   & v["drive"].notna() & v["gears"].notna() & v["cylinders"].notna())
    v["sample"] &= v["usable"]
    steps.append(("plausible weight and power (999 hp is a placeholder); engine, transmission and drive known", int(v["sample"].sum())))
    return v, steps


def features(v):
    f = v.copy()
    f["log_gpm"] = np.log(f["gpm"])
    f["log_weight"], f["log_hp"], f["log_displ"] = np.log(f["weight"]), np.log(f["hp"]), np.log(f["displ"])
    f["gears"] = np.where(f["trans"] == "CVT", 0, f["gears"])        # a CVT has no fixed gears; its own dummy carries it
    f["year"] = f["year"].astype(str)
    return f


def formula(extra=""):
    return f"log_gpm ~ {ATTRS}{extra} + C(year, Treatment('{FIRST}'))"


def year_effects(m, label):
    rows = [{"model": label, "year": FIRST, "pct": 0.0, "lo": 0.0, "hi": 0.0}]
    ci = m.conf_int()
    for y in range(FIRST + 1, LAST + 1):
        t = f"C(year, Treatment('{FIRST}'))[T.{y}]"
        rows.append({"model": label, "year": y, "pct": 100 * (np.exp(m.params[t]) - 1),
                     "lo": 100 * (np.exp(ci.loc[t, 0]) - 1), "hi": 100 * (np.exp(ci.loc[t, 1]) - 1)})
    return rows


def cv(f, fml, splitter, groups=None):
    pred = np.zeros(len(f))
    for tr, te in splitter.split(f, groups=groups):
        m = smf.ols(fml, f.iloc[tr]).fit()
        pred[te] = m.predict(f.iloc[te])
    err = f["log_gpm"] - pred
    return {"r2": float(1 - (err ** 2).sum() / ((f["log_gpm"] - f["log_gpm"].mean()) ** 2).sum()),
            "median_abs_pct_error": float(100 * np.median(np.abs(np.exp(err) - 1)))}


def decomposition(f, m):
    """Change in mean log consumption, first to last year = attribute changes x coefficients + technology (year effect).
    Exact for OLS with year dummies, because residuals average to zero within each year."""
    X = pd.DataFrame(m.model.exog, columns=m.model.exog_names, index=f.index)
    first, last = f["year"] == str(FIRST), f["year"] == str(LAST)
    parts = {}
    for name, cols in [("heavier", ["log_weight"]), ("more powerful", ["log_hp"]),
                       ("body and drive mix", [c for c in X.columns if c.startswith(("C(vtype", "C(drive"))])]:
        parts[name] = float(sum(m.params[c] * (X.loc[last, c].mean() - X.loc[first, c].mean()) for c in cols))
    parts["technology (year effect)"] = float(m.params[f"C(year, Treatment('{FIRST}'))[T.{LAST}]"])
    actual = float(f.loc[last, "log_gpm"].mean() - f.loc[first, "log_gpm"].mean())
    return {"actual_log_change": actual, "parts_log": parts, "sum_of_parts": sum(parts.values()),
            "actual_pct": 100 * (np.exp(actual) - 1), "parts_pct": {k: 100 * (np.exp(x) - 1) for k, x in parts.items()},
            "if_size_and_power_stayed_pct": 100 * (np.exp(parts["technology (year effect)"] + parts["body and drive mix"]) - 1),
            "share_of_technology_taken_back": -(parts["heavier"] + parts["more powerful"] + parts["body and drive mix"]) / parts["technology (year effect)"],
            "mean_weight": {FIRST: float(f.loc[first, "weight"].mean()), LAST: float(f.loc[last, "weight"].mean())},
            "mean_hp": {FIRST: float(f.loc[first, "hp"].mean()), LAST: float(f.loc[last, "hp"].mean())},
            "mean_gpm": {FIRST: float(f.loc[first, "gpm"].mean()), LAST: float(f.loc[last, "gpm"].mean())},
            "n": {FIRST: int(first.sum()), LAST: int(last.sum())}}


def physics(f):
    """City vs highway: does weight matter more in stop-and-go, and air drag more at speed? (separate from the decomposition)"""
    rows = []
    g = f[(f["coef_a"] > 0) & (f["coef_c"] > 0)].copy()
    g["log_a"], g["log_c"] = np.log(g["coef_a"]), np.log(g["coef_c"])
    for cycle in ["FTP", "HWY"]:
        g["y"] = np.log(g[cycle])
        m = smf.ols(f"y ~ log_weight + log_hp + log_c + log_a + C(year)", g).fit(cov_type="cluster", cov_kwds={"groups": g["group"].astype("category").cat.codes})
        for term in ["log_weight", "log_hp", "log_c", "log_a"]:
            rows.append({"cycle": "city" if cycle == "FTP" else "highway", "term": term, "elasticity": m.params[term],
                         "lo": m.conf_int().loc[term, 0], "hi": m.conf_int().loc[term, 1], "n": int(m.nobs)})
    X = g[["log_weight", "log_hp", "log_c", "log_a"]].assign(const=1).values
    vif = {c: float(variance_inflation_factor(X, i)) for i, c in enumerate(["log_weight", "log_hp", "log_c", "log_a"])}
    return pd.DataFrame(rows), vif


def main():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    a = load()
    v, steps = clean(a)
    (out / "cleaning.json").write_text(json.dumps(steps, indent=1) + "\n")
    v26, _ = clean(a, last=2026)                       # why 2026 is left out: same rules, compare its mix with 2025's
    s26 = v26[v26["sample"] & v26["year"].isin([2025, 2026])].groupby("year").agg(
        vehicles=("gpm", "size"), mean_weight=("weight", "mean"), mean_hp=("hp", "mean"), cvt_share=("trans", lambda x: (x == "CVT").mean()))
    (out / "partial_2026.json").write_text(json.dumps(s26.round(3).to_dict(orient="index"), indent=2) + "\n")
    for name, n in steps:
        print(f"{n:8,}  {name}")
    f = features(v[v["sample"]])
    clusters = f["group"].astype("category").cat.codes
    fit = lambda fml, d=f, g=clusters: smf.ols(fml, d).fit(cov_type="cluster", cov_kwds={"groups": g})

    raw = fit(f"log_gpm ~ C(year, Treatment('{FIRST}'))")
    attrs = fit(formula())
    tech = fit(formula(" + " + TECH))
    ye = pd.DataFrame(year_effects(raw, "raw average") + year_effects(attrs, "same weight, power, body, drive")
                      + year_effects(tech, "...and same engine and transmission"))
    ye.round(3).to_csv(out / "year_effects.csv", index=False)

    hc3 = smf.ols(formula(), f).fit(cov_type="HC3")
    coefs = []
    for label, m in [("attributes", attrs), ("attributes + technology", tech)]:
        for term in m.params.index:
            if term == "Intercept" or term.startswith("C(year"):
                continue
            coefs.append({"model": label, "term": term, "coef": m.params[term], "se_cluster": m.bse[term], "p_cluster": m.pvalues[term],
                          "se_hc3": hc3.bse.get(term, np.nan) if label == "attributes" else np.nan})
    coefs = pd.DataFrame(coefs)
    coefs.round(5).to_csv(out / "coefficients.csv", index=False)

    dec = decomposition(f, attrs)
    tech_share = 1 - tech.params[f"C(year, Treatment('{FIRST}'))[T.{LAST}]"] / attrs.params[f"C(year, Treatment('{FIRST}'))[T.{LAST}]"]
    dec.update({"r2_attributes": attrs.rsquared, "r2_technology": tech.rsquared, "n": int(attrs.nobs),
                "share_of_year_effect_explained_by_engine_and_transmission": tech_share,
                "clusters": int(f["group"].nunique()), "models": int(f["model_key"].nunique())})
    (out / "decomposition.json").write_text(json.dumps(dec, indent=2) + "\n")

    groups = f["model_key"]
    cvres = {name: {"random 10-fold": cv(f, fml, KFold(10, shuffle=True, random_state=0)),
                    "10-fold, whole models held out": cv(f, fml, GroupKFold(10), groups)}
             for name, fml in [("attributes", formula()), ("attributes + technology", formula(" + " + TECH))]}
    (out / "cv.json").write_text(json.dumps(cvres, indent=2) + "\n")

    rob = []
    allv = features(v[v["usable"]])
    for label, d, fml in [("main", f, formula()),
                          ("hybrids kept, with a hybrid dummy", allv, formula(" + hybrid")),
                          ("city test only", f.assign(log_gpm=np.log(f["FTP"])), formula()),
                          ("highway test only", f.assign(log_gpm=np.log(f["HWY"])), formula()),
                          ("each model counted once per year", f.groupby(["year", "model_key"], as_index=False).first(), formula())]:
        m = fit(fml, d, d["group"].astype("category").cat.codes)
        rob.append({"version": label, "n": int(m.nobs), "weight_elasticity": m.params["log_weight"], "hp_elasticity": m.params["log_hp"],
                    f"year_{LAST}_pct": 100 * (np.exp(m.params[f"C(year, Treatment('{FIRST}'))[T.{LAST}]"]) - 1)})
    rob = pd.DataFrame(rob)
    rob.round(4).to_csv(out / "robustness.csv", index=False)

    phys, vif = physics(f)
    phys.round(4).to_csv(out / "physics.csv", index=False)
    (out / "physics_vif.json").write_text(json.dumps(vif, indent=2) + "\n")
    f.to_csv(out / "model_data.csv", index=False)

    print(f"n {int(attrs.nobs)}, {dec['models']} models, {dec['clusters']} manufacturer clusters; R2 attributes {attrs.rsquared:.3f}, + technology {tech.rsquared:.3f}")
    print(coefs[coefs["model"] == "attributes"].round(4).to_string(index=False))
    print(ye[ye["year"] == LAST].round(1).to_string(index=False))
    print(json.dumps({k: dec[k] for k in ["actual_pct", "parts_pct", "if_size_and_power_stayed_pct", "mean_weight", "mean_hp", "mean_gpm", "n", "sum_of_parts", "actual_log_change"]}, indent=1))
    print(json.dumps(cvres, indent=1))
    print(rob.round(3).to_string(index=False))
    print(phys.round(3).to_string(index=False), vif)


if __name__ == "__main__":
    main()
