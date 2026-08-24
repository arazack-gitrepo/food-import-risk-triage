"""the dashboard. this is the demo an inspector would actually use.

    python run_experiment.py      # run this first, it trains the model
    streamlit run app/main.py

an inspector types in what is on the shipping paperwork, the model scores it,
and the screen turns that score into a decision: release it, take a closer
look, or open it. it also shows which fields pushed the score up or down, so
the decision is not just a number appearing from nowhere.

nothing here trains or tests anything. the model is trained by
run_experiment.py and just loaded from a file. if a number on this screen
disagrees with results/SUMMARY.md then something is out of date and the
pipeline needs rerunning.
"""
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rasff.config import LABEL_SCHEMES, MODELS_DIR, PREDS_DIR, RAW_CSV  # noqa: E402
from rasff.data.cleaning import clean_text_field, extract_hazard_category  # noqa: E402

MODEL_PATH = MODELS_DIR / "deployment_model.joblib"
MODEL_CARD = MODELS_DIR / "deployment_model_card.json"
STYLE_PATH = Path(__file__).with_name("style.css")

# the shipments the model was never trained on. everything on screen is ranked
# against these and nothing else. if the file is missing I want the app to stop
# dead rather than quietly fall back to the full dataset, because that would be
# comparing each shipment against ones the model already memorised.
TEST_PATH = PREDS_DIR / "test_window.csv"
DATA_PATH = RAW_CSV

SYSTEM_NAME = "Import Risk Triage"
SYSTEM_SUB = "Food and feed consignment screening"

LABELS = {
    "notification_type": "Type of notification",
    "product_category": "Product",
    "product_type": "Food or feed",
    "hazard_category": "Hazard",
    "origin_country": "Country of origin",
    "notifying_country": "Reporting country",
}

st.set_page_config(page_title=f"{SYSTEM_NAME} — screening terminal",
                   page_icon="▤", layout="wide")

CSS = f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>"
st.markdown(CSS, unsafe_allow_html=True)

# only needed if someone uploads a raw file that still has the hazard in curly
# brackets. the real version of this lives in rasff.data.cleaning.
BRACE = re.compile(r"\{([^}]*)\}")
# renames the portal's column names to the ones the model expects. same job as
# the version in config.py, trimmed to just the fields this screen uses.
RENAME = {"classification": "notification_type", "category": "product_category",
          "type": "product_type", "origin": "origin_country"}
# this must match the version in config.py. if it does not, the "% serious on
# record" figures on screen would be counting a different definition of serious
# than the model was trained on, and quietly disagree with it.
LABEL_MAP = {"serious": "serious", "potentially serious": "serious",
             "potential risk": "not_serious", "not serious": "not_serious",
             "no risk": "no_risk"}


def pretty(v):
    """tidy a value up for display."""
    s = re.sub(r"\s+", " ", str(v).replace("/", " / ").replace("_", " ")).strip()
    return s[:1].upper() + s[1:]


def tidy(s):
    """tidy a typed-in value so it matches what the model was trained on.

    if this did not match the tidying done during training, a perfectly valid
    country name typed here would not be recognised.
    """
    if pd.isna(s):
        return "unknown"
    return re.sub(r"\s+", " ", str(s).strip().lower()) or "unknown"


@st.cache_resource(show_spinner=False)
def load_model(path):
    """load the trained model and build the dropdown lists from it.

    the model already knows every country and product category it was trained
    on, so the dropdowns are read straight out of it rather than from a list I
    would have to keep up to date by hand. that way the screen can never offer
    an option the model has not seen.
    """
    model = joblib.load(path)
    ct = model.named_steps["features"]
    cols = list(ct.transformers_[0][2])
    ohe = ct.transformers_[0][1]
    raw = {c: [str(x) for x in v.tolist()] for c, v in zip(cols, ohe.categories_)}
    clean = {c: sorted(x for x in vals if "," not in str(x)) for c, vals in raw.items()}
    tolerant = getattr(ohe, "handle_unknown", "error") != "error"
    return model, cols, clean, raw, tolerant


def prepare(frame, cols):
    """get an uploaded file into the shape the model expects.

    tidies the column names, renames them, works out the hazard category if it
    is missing, and cleans up the values. also reports anything still missing.
    """
    df = frame.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
                  for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    if "hazard_category" not in df.columns and "hazards" in df.columns:
        # same function the pipeline trained with, imported not copied.
        df["hazard_category"] = (
            df["hazards"].apply(extract_hazard_category).fillna("unknown"))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, missing
    for c in cols:
        df[c] = df[c].apply(tidy)
    return df, []


@st.cache_data(show_spinner=False)
def history(_model, cols, path, test_path):
    """score all the held-out shipments once and keep them as a yardstick.

    everything on screen is "compared to what?", and this is the what. it
    gives back the scores to compare against, how often each country or
    product has actually been serious in the past, and how common each value
    is.

    the underscore on _model just stops streamlit trying to cache the model
    itself, which it cannot do.
    """
    # no fallback to `path` on purpose. see the note on TEST_PATH above.
    src = test_path
    if not os.path.exists(src):
        return None, None, None, None
    try:
        raw = pd.read_csv(src, engine="c", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        return None, None, None, None
    prepped, _ = prepare(raw, cols)
    if prepped is None or len(prepped) == 0:
        return None, None, None, None
    si = list(_model.classes_).index("serious")
    try:
        scores = np.sort(_model.predict_proba(prepped[cols])[:, si])
    except ValueError:
        return None, None, None, None
    freqs = {c: prepped[c].value_counts() for c in cols}
    rates = None
    if "risk_decision" in prepped.columns:
        h = prepped.copy()
        h["lab"] = h.risk_decision.map(LABEL_MAP)
        h = h[h.lab.notna()]
        if len(h):
            h["ser"] = (h.lab == "serious").astype(int)
            rates = {c: h.groupby(c).ser.agg(["mean", "size"]) for c in cols}
    return scores, rates, freqs, os.path.basename(src)


def rank_of(ref, score):
    """how this shipment compares to everything else, on a 0 to 100 scale.

    90 means riskier than 90% of the shipments on record.

    lots of shipments get exactly the same score, and the simple way of doing
    this would put every one of them at the bottom of its tied group, which
    would make them all look safer than they are. so I use the middle of the
    tie instead.
    """
    if ref is None or len(ref) == 0:
        return None
    lo = int(np.searchsorted(ref, score, side="left"))
    hi = int(np.searchsorted(ref, score, side="right"))
    return float((lo + hi) / 2.0 / len(ref) * 100.0)


def bands(budget):
    """work out where the green, amber and red bands sit.

    the red band is exactly as wide as the inspection budget, so if you can
    open 10% of shipments, the top 10% get flagged. that is deliberately the
    same rule the prioritisation analysis measures, which is why the 1.47x
    figure from my report actually applies to what this screen does.
    """
    stop = budget
    hold = min(budget, (100 - budget) * 0.35)
    return 100 - stop - hold, hold, stop


def verdict_for(rank, budget):
    """turn the ranking into the actual decision: release, review or inspect."""
    g, a, _ = bands(budget)
    if rank >= g + a:
        return "INSPECT", "var(--stop)", "Red channel"
    if rank >= g:
        return "REVIEW", "var(--hold)", "Amber channel"
    return "RELEASE", "var(--pass)", "Green channel"


def scale_html(rank, budget):
    """draws the coloured bar with the marker showing where this shipment sits."""
    g, a, r = bands(budget)
    pos = min(max(rank, 1.5), 98.5)   # keep the marker off the edges
    return f"""<div class="scale">
  <div class="pinrow"><span class="pin" style="left:{pos}%">&#9660;</span></div>
  <div class="bar3">
    <div class="sg" style="width:{g}%;background:var(--pass)"></div>
    <div class="sg" style="width:{a}%;background:var(--hold)"></div>
    <div class="sg hatch" style="width:{r}%;background:var(--stop)"></div>
  </div>
  <div class="axis">
    <span class="l">Lower risk</span>
    <span class="r">Higher risk</span>
  </div>
</div>"""


def weighted_median(values, weights):
    """the middle value, weighted by how often each one actually turns up.

    without the weighting, "typical" would mean the middle of a list of
    options, most of which almost never arrive. this makes it mean a typical
    real shipment instead.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return float(np.median(v))
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    return float(v[min(int(np.searchsorted(cum, total / 2.0)), len(v) - 1)])


def drivers(model, cols, picked, si, cats, freqs):
    """work out why this particular shipment scored what it did.

    for each field, I re-score the same shipment with every other value that
    field could have had, then compare the actual value against a typical one.
    the difference is what that field contributed.

    so "origin: Pakistan, +0.12" means being from Pakistan pushed this
    shipment's score up by 0.12 compared to a typical origin.

    this explains ONE shipment. it is a different thing from the overall
    feature importance in the ablation code, which is about the model as a
    whole. do not quote one as if it were the other.
    """
    rows, meta = [], []
    for c in cols:
        options = list(dict.fromkeys(list(cats.get(c, [])) + [picked[c]]))
        for v in options:
            r = dict(picked)
            r[c] = v
            rows.append(r)
            meta.append((c, v))
    scored = model.predict_proba(pd.DataFrame(rows)[cols])[:, si]
    frame = pd.DataFrame(meta, columns=["field", "value"])
    frame["score"] = scored

    out = []
    for c in cols:
        sub = frame[frame.field == c]
        if freqs is not None and c in freqs:
            counts = freqs[c]
            typical = weighted_median(sub.score.values,
                                      [float(counts.get(v, 0.0)) for v in sub.value])
        else:
            typical = float(sub.score.median())
        chosen = float(sub.loc[sub.value == picked[c], "score"].iloc[0])
        best = sub.loc[sub.score.idxmin()]
        out.append(dict(field=c, effect=chosen - typical,
                        lowest_value=best.value, lowest_score=float(best.score)))
    return pd.DataFrame(out).sort_values("effect", key=abs, ascending=False)


def finding_html(n, row, picked, rates):
    """draw one line of the explanation panel.

    bar to the right means this field pushed the score up, left means down.

    the "% serious on record" note only shows when there are at least 30 past
    shipments behind it. below that the figure bounces around too much to put
    in front of an inspector as though it were solid.
    """
    eff = row.effect
    mag = min(abs(eff) / 0.35, 1.0) * 50
    if eff > 0.01:
        colour, left, amt = "var(--stop)", 50, f"+{eff:.2f}"
    elif eff < -0.01:
        colour, left, amt = "var(--pass)", 50 - mag, f"{eff:.2f}"
    else:
        colour, left, amt = "var(--soft)", 49.5, "0.00"

    extra = ""
    if rates is not None and row.field in rates:
        t = rates[row.field]
        v = picked[row.field]
        if v in t.index and t.loc[v, "size"] >= 30:
            extra = (f' <span style="color:var(--soft);font-weight:400">'
                     f'&middot; {t.loc[v,"mean"]*100:.0f}% serious on record</span>')

    return (f'<div class="fnd">'
            f'<div class="ix">{n}</div>'
            f'<div class="nm">{LABELS.get(row.field, row.field)}</div>'
            f'<div class="vl">{pretty(picked[row.field])}{extra}</div>'
            f'<div class="tr"><div class="md"></div>'
            f'<div class="fl" style="left:{left}%;width:{max(mag,1)}%;'
            f'background:{colour}"></div></div>'
            f'<div class="am" style="color:{colour}">{amt}</div>'
            f'</div>')


# the app starts here. everything above is helper functions.

# both files are produced by run_experiment.py. if either is missing I stop the
# app rather than carrying on, because a dashboard quietly scoring against the
# wrong data still looks like it is working perfectly.
if not os.path.exists(TEST_PATH):
    st.error(
        f"held-out test window not found at {TEST_PATH}. "
        "run `python run_experiment.py` first. this app will not score against "
        "the full export, that would rank consignments among rows the model "
        "trained on."
    )
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"model file {MODEL_PATH} not found. run `python run_experiment.py`.")
    st.stop()

model, COLS, CATS, RAWCATS, TOLERANT = load_model(MODEL_PATH)

# which of the model's outputs is the "serious" one. the same lookup the
# prioritisation code does.
SI = list(model.classes_).index("serious")
REF, RATES, FREQS, REF_NAME = history(model, COLS, DATA_PATH, TEST_PATH)
NOW = datetime.now()

items = [f'<div class="it">Reference set <b>{REF_NAME or "not loaded"}</b></div>']
if REF is not None:
    items.append(f'<div class="it">Records on file <b>{len(REF):,}</b></div>')

st.markdown(
    f'<div class="mast"><div><span class="nm">{SYSTEM_NAME}</span>'
    f'<span class="sb">{SYSTEM_SUB}</span></div>'
    f'<div class="rg">{NOW:%d %b %Y &middot; %H:%M}</div></div>'
    f'<div class="strip">{"".join(items)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- rendering


def render_determination(picked, budget, score, rank):
    """draw the whole result screen: the decision, the scale, the numbers, and
    the explanation underneath.

    used for both a typed-in shipment and a row from an uploaded file, so the
    two can never end up showing different things.
    """
    name, colour, channel = verdict_for(rank, budget)
    st.markdown(
        f'<div class="dock" style="--chan:{colour}">'
        f'<div class="bar"><span class="lb">Screening determination</span></div>'
        f'<div class="bd">'
        f'<div class="desig"><div class="stamp">{name}</div>'
        f'<div class="gl">{channel}<br><b>Riskier than {rank:.0f}% of arrivals '
        f'on record</b></div></div>'
        f'{scale_html(rank, budget)}'
        f'<div class="figs">'
        f'<div class="f"><div class="k">Percentile rank</div>'
        f'<div class="v">{rank:.0f}<span class="u">%</span></div></div>'
        f'<div class="f"><div class="k">Threshold at current capacity</div>'
        f'<div class="v">{100-budget:.0f}<span class="u">%</span></div></div>'
        f'<div class="f"><div class="k">Model score</div>'
        f'<div class="v">{score:.3f}</div></div>'
        f'</div></div></div>', unsafe_allow_html=True)

    drv = drivers(model, COLS, picked, SI, CATS, FREQS)
    st.markdown('<div class="head" style="margin-top:24px">Basis of determination'
                '</div>', unsafe_allow_html=True)
    st.markdown("".join(finding_html(i, r, picked, RATES)
                        for i, r in enumerate(drv.itertuples(), 1)),
                unsafe_allow_html=True)
    st.markdown('<p class="tiny">Bars show each field&rsquo;s contribution relative '
                'to a typical arrival. Right raises the determination, left lowers '
                'it.</p>', unsafe_allow_html=True)

    top = drv.iloc[0]
    if abs(top.effect) > 0.02 and top.lowest_score < score - 0.05:
        st.markdown(
            f'<p class="tiny">Recorded as <b>{pretty(top.lowest_value)}</b> instead, '
            f'this consignment would score {top.lowest_score:.2f}.</p>',
            unsafe_allow_html=True)


# Row selection in st.dataframe landed in Streamlit 1.35. Fall back to a
# position box on older builds rather than failing at import time.
try:
    ROW_SELECT = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 35)
except Exception:
    ROW_SELECT = False


tab_manifest, tab_single = st.tabs(["Screen a manifest", "Check one consignment"])

with tab_manifest:
    up = st.file_uploader("Manifest file (CSV)", type=["csv"])

    if up is None:
        st.markdown('<p class="tiny">Upload a consignment file to rank it. '
                    'Select any row afterwards to see why it was placed there.</p>',
                    unsafe_allow_html=True)
    else:
        try:
            raw = pd.read_csv(up, engine="c", encoding="utf-8", on_bad_lines="skip")
        except Exception as exc:
            st.error(f"File could not be read: {exc}")
            st.stop()

        prepped, missing = prepare(raw, COLS)
        if prepped is None:
            st.error("Required fields absent: "
                     + ", ".join(LABELS.get(m, m) for m in missing))
            st.stop()
        if len(prepped) == 0:
            st.error("File contains no readable records.")
            st.stop()

        dropped = len(raw) - len(prepped)
        if dropped > 0:
            st.warning(f"{dropped:,} malformed record(s) skipped.")

        unknown = {c: sorted(set(prepped[c]) - set(RAWCATS.get(c, []))) for c in COLS}
        unknown = {c: v for c, v in unknown.items() if v}
        if unknown:
            if TOLERANT:
                affected = int(np.any(
                    [prepped[c].isin(v).values for c, v in unknown.items()], axis=0
                ).sum())
                fields = ", ".join(LABELS.get(c, c).lower() for c in unknown)
                st.caption(f"{affected:,} of {len(prepped):,} records contain "
                           f"combinations not present in the training data "
                           f"({fields}); those fields carry no signal for them.")
            else:
                detail = "; ".join(f"{LABELS.get(c, c)}: " + ", ".join(v[:4])
                                   + (f" +{len(v)-4}" if len(v) > 4 else "")
                                   for c, v in unknown.items())
                st.error("Model does not accept unseen categories — " + detail)
                st.stop()

        try:
            s = model.predict_proba(prepped[COLS])[:, SI]
        except ValueError as exc:
            st.error(f"Scoring failed: {exc}")
            st.stop()

        out = prepped[COLS].copy()
        out.insert(0, "risk", s.round(4))
        for c in reversed(("reference", "subject", "date")):
            if c in prepped.columns:
                out.insert(1, c, prepped[c].values)
        out = out.sort_values("risk", ascending=False).reset_index(drop=True)
        out.insert(0, "position", np.arange(1, len(out) + 1))

        b2 = st.slider("Inspection capacity", 1, 50, 10, 1, format="%d%%", key="b2")
        k = max(1, min(len(out), int(round(len(out) * b2 / 100))))
        out["action"] = np.where(np.arange(len(out)) < k, "INSPECT", "release")

        # Outcomes are only present when scoring a historical file. An
        # operational manifest has none, so the scorecard below stays hidden.
        ordered, total = None, 0
        if "risk_decision" in prepped.columns:
            ser = (prepped.risk_decision.map(LABEL_MAP) == "serious").astype(int).values
            total = int(ser.sum())
            if total:
                ordered = ser[np.argsort(-s)]

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="cell"><div class="n">{len(out):,}</div>'
                    f'<div class="l">consignments screened</div></div>',
                    unsafe_allow_html=True)
        c2.markdown(f'<div class="cell"><div class="n">{k:,}</div>'
                    f'<div class="l">referred for inspection</div></div>',
                    unsafe_allow_html=True)
        if ordered is not None:
            got = int(ordered[:k].sum())
            exp = k / len(out) * total
            c3.markdown(f'<div class="cell"><div class="n" style="color:var(--pass)">'
                        f'{got:,}</div><div class="l">serious cases in referrals, '
                        f'against {exp:.0f} without ranking &middot; '
                        f'{got/k*100:.1f}% precision</div></div>',
                        unsafe_allow_html=True)
        else:
            c3.markdown(f'<div class="cell"><div class="n">{out.risk.iloc[k-1]:.3f}'
                        f'</div><div class="l">score at referral threshold</div></div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="head" style="margin-top:20px">Referral list, '
                    'highest risk first</div>', unsafe_allow_html=True)

        cfg = {"risk": st.column_config.ProgressColumn(
            "Risk", min_value=0.0, max_value=1.0, format="%.3f")}
        chosen = None
        if ROW_SELECT:
            ev = st.dataframe(out, use_container_width=True, hide_index=True,
                              height=380, column_config=cfg,
                              on_select="rerun", selection_mode="single-row",
                              key="queue")
            picks = ev.selection["rows"] if ev and "selection" in ev else []
            if picks:
                chosen = int(picks[0])
        else:
            st.dataframe(out, use_container_width=True, hide_index=True,
                         height=380, column_config=cfg)
            pos = st.number_input("Show the basis for position", 1, len(out), 1, 1,
                                  key="posbox")
            chosen = int(pos) - 1

        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button("Download referral list", buf.getvalue(),
                           f"referral_list_{NOW:%Y%m%d_%H%M}.csv", "text/csv")

        if chosen is None:
            st.markdown('<p class="tiny">Select a row to see why it was placed '
                        'there.</p>', unsafe_allow_html=True)
        else:
            rec = out.iloc[chosen]
            rpicked = {c: rec[c] for c in COLS}
            rscore = float(rec["risk"])
            rrank = rank_of(REF, rscore)
            if rrank is None:
                rrank = rscore * 100
            label = rec["reference"] if "reference" in out.columns else ""
            st.markdown(f'<div class="head" style="margin-top:22px">Position '
                        f'{int(rec["position"])} of {len(out):,}'
                        f'{" &middot; " + str(label) if label else ""}</div>',
                        unsafe_allow_html=True)
            render_determination(rpicked, b2, rscore, rrank)

        if ordered is not None:
            cum = np.cumsum(ordered)
            pts = np.arange(1, 101)
            # same rounded cut-off for every line, or the ceiling can fall
            # fractionally below the model at small budgets
            ks = np.clip((len(ordered) * pts / 100).round().astype(int),
                         1, len(ordered))
            curve = pd.DataFrame({
                "Inspected %": pts,
                "This ranking": cum[ks - 1] / total * 100,
                "Without ranking": ks / len(ordered) * 100,
                "Theoretical maximum": np.minimum(ks, total) / total * 100,
            }).set_index("Inspected %")
            st.markdown('<div class="head" style="margin-top:24px">Serious cases '
                        'identified, by inspection capacity (%) &mdash; this file '
                        'records outcomes</div>', unsafe_allow_html=True)
            st.line_chart(curve, height=250,
                          color=["#1F3A54", "#98A2AA", "#1B5A40"])

with tab_single:
    left, right = st.columns([0.85, 1.45], gap="large")

    with left:
        st.markdown('<div class="head">Consignment particulars</div>',
                    unsafe_allow_html=True)
        defaults = {
            "notification_type": "alert notification",
            "product_category": "fruits and vegetables",
            "product_type": "food",
            "hazard_category": "pesticide residues",
            "origin_country": "turkiye",
            "notifying_country": "germany",
        }
        picked = {}
        for c in COLS:
            opts = CATS[c]
            d = defaults.get(c)
            ix = opts.index(d) if d in opts else 0
            picked[c] = st.selectbox(LABELS.get(c, c), opts, index=ix,
                                     format_func=pretty, key=f"s_{c}")

        st.markdown('<div class="head" style="margin-top:22px">Inspection capacity</div>',
                    unsafe_allow_html=True)
        budget = st.slider("Share of arrivals that can be inspected", 1, 50, 10, 1,
                           format="%d%%")

    score = float(model.predict_proba(pd.DataFrame([picked])[COLS])[0][SI])
    rank = rank_of(REF, score)
    if rank is None:
        rank = score * 100

    with right:
        render_determination(picked, budget, score, rank)
