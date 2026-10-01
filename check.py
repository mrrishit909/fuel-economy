"""The project's check: fails loudly if the cleaning, the model or the decomposition breaks.

    ./venv/bin/python check.py

1. the cleaning funnel recomputes from the raw files, and the sample has no carry-overs, hybrids or placeholder horsepower
2. an independent NumPy least-squares refit matches the model's weight, power and 2025 coefficients
3. the decomposition adds up exactly to the actual change in mean log fuel use, recomputed from the data
4. clustered standard errors are wider than heteroskedasticity-robust ones; robustness versions keep every sign
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis as A

HERE = Path(__file__).parent
R = HERE / "results"


def main():
    v, steps = A.clean(A.load())
    assert [list(s) for s in steps] == json.loads((R / "cleaning.json").read_text()), "funnel changed"
    s = v[v["sample"]]
    assert s.duplicated(["year", "Test Vehicle ID", "Test Veh Configuration #"]).sum() == 0
    assert s["year"].between(A.FIRST, A.LAST).all() and (s["hp"] != 999).all() and (s["ratio"] >= 1.2).all()
    assert not s["model"].astype(str).str.upper().str.contains(A.HYBRID_NAME).any()
    assert np.allclose(s["gpm"], 0.55 * s["FTP"] + 0.45 * s["HWY"])        # combined in gallons per 100 miles, not averaged MPG

    f = pd.read_csv(R / "model_data.csv")
    assert len(f) == steps[-1][1]
    X = pd.get_dummies(f[["vtype", "drive", "year"]].astype(str), dtype=float)
    X = X.drop(columns=["vtype_Car", "drive_front", f"year_{A.FIRST}"])
    X.insert(0, "log_hp", f["log_hp"])
    X.insert(0, "log_weight", f["log_weight"])
    X.insert(0, "const", 1.0)
    beta = dict(zip(X.columns, np.linalg.lstsq(X.values, f["log_gpm"].values, rcond=None)[0]))
    c = pd.read_csv(R / "coefficients.csv").query("model == 'attributes'").set_index("term")["coef"]
    d = json.loads((R / "decomposition.json").read_text())
    assert abs(beta["log_weight"] - c["log_weight"]) < 1e-4 and abs(beta["log_hp"] - c["log_hp"]) < 1e-4
    assert abs(beta[f"year_{A.LAST}"] - d["parts_log"]["technology (year effect)"]) < 1e-6

    m = f.groupby("year")["log_gpm"].mean()
    actual = m[A.LAST] - m[A.FIRST]
    assert abs(actual - d["actual_log_change"]) < 1e-9 and abs(sum(d["parts_log"].values()) - actual) < 1e-9
    w = f.groupby("year")[["log_weight", "log_hp"]].mean()
    assert abs(beta["log_weight"] * (w.loc[A.LAST, "log_weight"] - w.loc[A.FIRST, "log_weight"]) - d["parts_log"]["heavier"]) < 1e-6

    cc = pd.read_csv(R / "coefficients.csv").query("model == 'attributes'").set_index("term")
    assert (cc.loc[["log_weight", "log_hp"], "se_cluster"] > cc.loc[["log_weight", "log_hp"], "se_hc3"]).all()
    rob = pd.read_csv(R / "robustness.csv")
    assert (rob["weight_elasticity"] > 0).all() and (rob["hp_elasticity"] > 0).all() and (rob[f"year_{A.LAST}_pct"] < 0).all()
    p = pd.read_csv(R / "physics.csv").set_index(["cycle", "term"])["elasticity"]
    assert p["highway", "log_c"] > p["city", "log_c"] > 0

    print(f"OK: funnel recomputes ({len(f):,} vehicles, no carry-overs, hybrids or 999-hp placeholders); NumPy refit matches "
          f"(weight {beta['log_weight']:.3f}, power {beta['log_hp']:.3f}); technology {d['parts_pct']['technology (year effect)']:+.1f}% "
          f"and bigger cars' {100 * d['share_of_technology_taken_back']:.0f}% give-back add up to the actual {d['actual_pct']:+.1f}%; "
          f"signs hold in all {len(rob)} versions")


if __name__ == "__main__":
    main()
