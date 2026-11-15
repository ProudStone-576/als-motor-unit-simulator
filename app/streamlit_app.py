"""Interactive explorer for mndsim.

Run with::

    pip install "mndsim[app]"      # or: pip install -e ".[app]"
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

import mndsim
from mndsim import plotting
from mndsim.phenotypes import PHENOTYPES
from mndsim.regions import REGIONS
from mndsim.treatment import TREATMENTS

st.set_page_config(page_title="mndsim - motor unit degeneration simulator", layout="wide")

DISCLAIMER = (
    "Research and teaching tool. Parameters are illustrative and calibrated only to population-level "
    "summaries; nothing here is a prediction for a real person or medical advice."
)

PARAM_SPECS = {
    "lmn_rate": ("LMN degeneration hazard (/month)", 0.0, 0.4, 0.005),
    "umn_rate": ("UMN loss rate (/month)", 0.0, 0.1, 0.002),
    "excitotoxicity": ("Excitotoxic amplification", 0.0, 2.0, 0.05),
    "spread_rate": ("Spread rate (neuraxis units/month)", 0.0, 0.2, 0.005),
    "sprouting_rate": ("Sprouting rate (/month)", 0.1, 6.0, 0.1),
    "max_enlargement": ("Max unit enlargement (x)", 1.0, 10.0, 0.25),
}


def patient_sidebar() -> mndsim.Patient:
    st.sidebar.header("Virtual patient")
    name = st.sidebar.selectbox("Phenotype preset", list(PHENOTYPES), index=0)
    base = PHENOTYPES[name]
    onset = st.sidebar.selectbox("Region of onset", REGIONS, index=REGIONS.index(base.onset_region))
    changes = {"onset_region": onset}
    with st.sidebar.expander("Disease parameters", expanded=False):
        for key, (label, lo, hi, step) in PARAM_SPECS.items():
            changes[key] = st.slider(label, lo, hi, float(getattr(base, key)), step, key=f"{name}-{key}")
    return base.replace(**changes)


@st.cache_data(show_spinner=False)
def run(patient_json: str, tx_names: tuple[str, ...], start: float, months: float, stochastic: bool):
    patient = mndsim.Patient.from_dict(json.loads(patient_json))
    txs = [TREATMENTS[n].at(start) for n in tx_names]
    return mndsim.simulate(
        patient, txs or None, months=months, stochastic=stochastic, seed=7, record_units=True
    )


def patient_key(p: mndsim.Patient) -> str:
    """Hashable, cache-friendly representation of a patient."""
    return json.dumps(p.to_dict(), sort_keys=True)


def tab_patient(patient: mndsim.Patient) -> None:
    months = st.slider("Months to simulate (from disease onset)", 24, 180, 96, 6)
    base = run(patient_key(patient), (), 0.0, float(months), False)
    onset = base.symptom_onset[0]
    c1, c2 = st.columns([2, 1])
    with c2:
        tx_names = st.multiselect("Treatment(s)", [n for n in TREATMENTS if n != "placebo"])
        start_after = st.slider("Start, months after symptom onset", -12.0, 48.0, 12.0, 1.0)
        stochastic = st.toggle("Stochastic motor-unit fates", value=False)
    start = (onset if np.isfinite(onset) else 0.0) + start_after
    res = run(patient_key(patient), tuple(tx_names), float(start), float(months), stochastic)
    s = res.summary().iloc[0]
    with c1:
        cols = st.columns(4)
        cols[0].metric(
            "Symptom onset", f"{s.symptom_onset_months:.1f} mo", help="Months after biological disease onset"
        )
        cols[1].metric(
            "Units lost by then",
            f"{100 * s.mu_loss_at_symptom_onset:.0f}%",
            help="Motor units already lost in the region of onset when weakness is first noticed",
        )
        surv = s.survival_from_symptom_onset
        base_surv = base.summary().iloc[0].survival_from_symptom_onset
        delta = None if not tx_names or np.isnan(surv) or np.isnan(base_surv) else f"{surv - base_surv:+.1f}"
        cols[2].metric(
            "Survival",
            "beyond horizon" if np.isnan(surv) else f"{surv:.1f} mo",
            delta,
            help="Months from symptom onset to respiratory failure (death or permanent ventilation)",
        )
        cols[3].metric(
            "Year-1 slope",
            f"{s.alsfrs_slope_first_year:.2f}/mo",
            help="Functional score change per month over the first year after symptom onset",
        )
    st.pyplot(res.plot(), use_container_width=True)
    with st.expander("Motor-unit raster (region of onset)"):
        st.pyplot(plotting.plot_motor_units(res), use_container_width=True)
    st.download_button("Download trajectory (CSV)", res.to_frame().to_csv(index=False), "trajectory.csv")


def tab_counterfactual(patient: mndsim.Patient) -> None:
    st.markdown("Same patient, different treatments, same start time.")
    months = 120.0
    base = run(patient_key(patient), (), 0.0, months, False)
    onset = base.symptom_onset[0]
    if not np.isfinite(onset):
        st.info("This patient never becomes symptomatic within the horizon.")
        return
    chosen = st.multiselect(
        "Treatments", list(TREATMENTS)[1:], default=["neuroprotective", "nmj_stabilizer", "symptomatic"]
    )
    start_after = st.slider("Start, months after symptom onset", 0.0, 36.0, 12.0, 1.0, key="cf_start")
    metric = st.radio(
        "Show",
        ["alsfrs", "fvc"],
        horizontal=True,
        format_func={"alsfrs": "Functional score", "fvc": "FVC"}.get,
    )
    results = {"untreated": base}
    for name in chosen:
        results[name] = run(patient_key(patient), (name,), float(onset + start_after), months, False)
    fig = plotting.plot_trajectories(results, metric, marks={"start": start_after})
    st.pyplot(fig, use_container_width=True)
    rows = [{"arm": k, "event (months after onset)": v.event_time[0] - onset} for k, v in results.items()]
    df = pd.DataFrame(rows)
    df["gain vs untreated"] = df.iloc[:, 1] - df.iloc[0, 1]
    st.dataframe(df.round(1), hide_index=True)


@st.cache_data(show_spinner=False)
def trial_power(tx_name: str, n_max: int, duration: float, min_dfs: float, genotype: str, seed: int):
    design = mndsim.TrialDesign(
        n_per_arm=n_max, duration=duration, min_delta_fs=min_dfs, genotype=genotype or None
    )
    pool_size = max(1500, 6 * n_max)
    pool = mndsim.build_trial_pool(TREATMENTS[tx_name], design, pool_size=pool_size, seed=seed)
    sizes = sorted({max(10, n_max // 6), n_max // 3, n_max // 2, (2 * n_max) // 3, n_max})
    curve = mndsim.power_curve(pool, sizes, n_trials=150, seed=seed + 1)
    return curve, pool.individual_effects(), pool.screen_failure_rate


def tab_trial() -> None:
    st.markdown(
        "Every eligible virtual patient is simulated **both** with and without treatment; replicate "
        "trials resample the pool, add rater noise and dropout, and test five endpoints."
    )
    c = st.columns(5)
    tx = c[0].selectbox("Treatment", list(TREATMENTS), index=list(TREATMENTS).index("anti_excitotoxic"))
    n_max = c[1].select_slider("Max patients per arm", [60, 120, 180, 240, 300], value=180)
    duration = c[2].select_slider("Duration (months)", [6, 9, 12, 18], value=12)
    min_dfs = c[3].slider("Enrich: min progression rate", 0.0, 1.5, 0.0, 0.1)
    genotype = c[4].selectbox("Genotype", ["", "SOD1", "C9orf72"], format_func=lambda g: g or "all")
    if st.button("Run power analysis", type="primary"):
        with st.spinner("Screening virtual patients and simulating both arms..."):
            curve, effects, sfr = trial_power(tx, n_max, float(duration), min_dfs, genotype, 0)
        st.caption(f"Screen failure rate: {100 * sfr:.0f}%")
        st.pyplot(plotting.plot_power_curve(curve), use_container_width=True)
        st.dataframe(curve.pivot(index="n_per_arm", columns="endpoint", values="power").round(2))
        st.markdown("**Individual treatment effects** (treated minus untreated, same patient)")
        st.dataframe(effects[["alsfrs_preserved", "event_delay"]].describe().round(2).T)
        n_events = int(effects.event_control.notna().sum())
        st.caption(
            f"{int(effects.event_prevented.sum())} of {n_events} respiratory-failure events in the untreated "
            "arm were prevented during follow-up; `event_delay` covers the rest."
        )


def main() -> None:
    st.title("Motor unit degeneration simulator")
    st.caption(DISCLAIMER)
    patient = patient_sidebar()
    t1, t2, t3, t4 = st.tabs(["Patient", "Counterfactuals", "Virtual trial", "About the model"])
    with t1:
        tab_patient(patient)
    with t2:
        tab_counterfactual(patient)
    with t3:
        tab_trial()
    with t4:
        st.markdown(
            "Each of four regions (bulbar, respiratory, upper and lower limb) has a pool of 120 motor "
            "units organised by the size principle. Units move from **healthy** to **denervating** "
            "(dying-back) to **lost**; surviving units adopt orphaned fibres by collateral sprouting, "
            "which hides loss until compensation is exhausted. Upper motor neuron loss lowers maximal "
            "drive and changes excitotoxic stress. Disease spreads contiguously along the neuraxis. "
            "See `docs/MODEL.md` in the repository for equations, parameters and references."
        )


main()
