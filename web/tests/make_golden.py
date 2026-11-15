"""Write golden trajectories from the Python engine for the JavaScript parity test.

python web/tests/make_golden.py
node --test web/tests/parity.test.mjs
"""

import json
import math
import pathlib

import numpy as np

import mndsim
from mndsim.engine import _unit_vulnerability
from mndsim.phenotypes import PHENOTYPES
from mndsim.pool import MotorUnitPool

OUT = pathlib.Path(__file__).with_name("golden.json")


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def tx_dict(t):
    return {k: clean(v) for k, v in t.to_dict().items()}


def arr(a):
    return [None if not np.isfinite(v) else float(f"{v:.10g}") for v in np.asarray(a, dtype=float).ravel()]


CASES = [
    ("untreated phenotypes", [PHENOTYPES[k] for k in PHENOTYPES], [], {}),
    (
        "combination with stop and washout",
        [PHENOTYPES["typical"], PHENOTYPES["bulbar_onset"]],
        [
            mndsim.TREATMENTS["neuroprotective"].at(14.0, stop=40.0),
            mndsim.TREATMENTS["anti_excitotoxic"].at(20.0),
            mndsim.TREATMENTS["symptomatic"].at(30.0),
        ],
        {},
    ),
    (
        "gene-targeted, rescue, sprouting with time origin",
        [PHENOTYPES["sod1_familial"], PHENOTYPES["typical"]],
        [
            mndsim.TREATMENTS["gene_targeted"],
            mndsim.TREATMENTS["nmj_stabilizer"],
            mndsim.TREATMENTS["sprouting_enhancer"],
        ],
        {"time_origin": [6.0, 18.0], "dt": 0.5, "record_every": 0.5},
    ),
    (
        "cohort sample",
        mndsim.sample_cohort(6, seed=11),
        [],
        {"months": 120.0},
    ),
]


def main():
    pool = MotorUnitPool.from_config()
    cases = []
    for name, patients, txs, kw in CASES:
        months = kw.get("months", 96.0)
        dt = kw.get("dt", 0.25)
        rec = kw.get("record_every", 1.0)
        origin = kw.get("time_origin", 0.0)
        res = mndsim.simulate(
            patients, txs, months=months, dt=dt, record_every=rec, time_origin=np.asarray(origin)
        )
        vul = _unit_vulnerability(patients, 0, pool, mndsim.DEFAULT_CONFIG, 0)
        pats = []
        for p in patients:
            d = p.to_dict()
            d["region_severity"] = list(p.region_severity)
            pats.append({k: clean(v) for k, v in d.items()})
        cases.append(
            {
                "name": name,
                "patients": pats,
                "treatments": [tx_dict(t) for t in txs],
                "options": {"months": months, "dt": dt, "recordEvery": rec, "timeOrigin": origin},
                "vulnerability": [arr(v) for v in vul],
                "expected": {
                    "time": arr(res.time),
                    "alsfrs": arr(res.alsfrs),
                    "fvc": arr(res.fvc),
                    "nfl": arr(res.nfl),
                    "mvc": arr(res.mvc),
                    "muCount": arr(res.mu_count),
                    "cmap": arr(res.cmap),
                    "enlargement": arr(res.enlargement),
                    "symptomOnset": arr(res.symptom_onset),
                    "symptomRegion": [int(x) for x in res.symptom_region],
                    "eventTime": arr(res.event_time),
                },
            }
        )
    from scipy import stats

    t_values = [
        [t, df, float(2 * stats.t.sf(abs(t), df))]
        for t, df in [(0.5, 3), (2.1, 10), (-3.3, 57.4), (1.96, 400), (4.0, 7.5)]
    ]
    OUT.write_text(json.dumps({"cases": cases, "t_test": t_values}))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
