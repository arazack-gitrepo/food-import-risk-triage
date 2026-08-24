# Food Import Risk Triage

MSc Data Science dissertation, University of Birmingham Dubai.

Predicts how risky an imported food shipment is, and ranks shipments so an
inspector knows which ones to open first. Built on RASFF, the EU's Rapid Alert
System for Food and Feed: 15,331 notifications from 2023 onward.

## Running it

```bash
pip install -r requirements.txt
python run_experiment.py
```

You need the data first. It isn't in this repo (see below). Download the RASFF
Window export, filter to 2022-01-01 through 2025-12-31, and save it as
`data/RASFF_window.csv`.

Download from 2022 even though the model only trains on 2023 onward. The 2022
rows are what show the labelling scheme changed, which is the justification for
the cut.

Filters: 2022-01-01 to 2025-12-31, everything else left unrestricted. That
should give you 19,890 rows. `run_experiment.py` prints the count at step 1 —
if it doesn't match, the filters are wrong.

That one command regenerates everything in `results/`. Takes about 10 to 15
minutes on a normal laptop. `--quick` runs a shorter version if you just want
to see it work.

To open the dashboard afterwards:

```bash
streamlit run app/main.py
```

If `streamlit` isn't on your PATH, `python -m streamlit run app/main.py` does
the same thing.

The dashboard works straight after cloning — the trained model and the
held-out test window are the only generated files committed to this repo, so
you don't need the RASFF export just to look at it. You do need it to
regenerate the results.

## What I found

The model scores 0.839 macro F1 using the full RASFF record. But most of that
comes from fields the inspector doesn't have yet when they're deciding what to
open:

| what the model knows | macro F1 | step |
|---|---|---|
| just the customs declaration | 0.686 | — |
| + notifying country | 0.699 | +0.012, could be noise |
| + notification type (a judgement the filing official already made) | 0.810 | **+0.111** |
| + hazard category (usually a lab result) | 0.839 | +0.029 |
| + post-handling fields | 0.829 | −0.010, significantly worse |

So 0.839 is what's achievable with the complete record, and **0.686 is what's
achievable before anything is opened**. Both are worth reporting. Quoting only
the first overstates what the model knows at decision time.

Each step is measured against the one directly below it with a paired
bootstrap, so the +0.111 is a tested difference rather than two point estimates
subtracted. Every tier is fitted with the same estimator, so a step reflects
the information added rather than a change of algorithm.

The last row is worth noting: fields recorded after the consignment has been
handled make the model measurably worse, not better. More information is not
automatically more signal.

**The written description depends entirely on what you compare it against.**
Against the full record it looks useless. Against the information actually
available before inspection, it helps:

| | macro F1 | vs baseline |
|---|---|---|
| text vs the customs declaration alone | 0.772 | **+0.086**, significantly better |
| text vs the full record | 0.772 | −0.066, significantly worse |

An earlier version of this work concluded that text adds nothing. That
conclusion was an artefact of benchmarking against a baseline that already
contained the lab result. `scripts/reconcile_baseline.py` runs both comparisons
side by side.

The useful part is the ranking. At a 10% inspection budget the deployable
model catches 14.7% of the serious cases, against a ceiling of 16.2% (you
can't catch more than you can open) and 14.0% for the origin-country rule of
thumb inspectors already use. That's 91% of what perfect foresight would get,
using nothing but the shipping paperwork.

## Where things are

`WHERE_THINGS_ARE.md` maps every number in my dissertation to the function
that computes it.

The short version: all scoring lives in `src/rasff/evaluation/metrics.py`, and
the train/tune/test protocol is one function, `fit_score()`, in
`src/rasff/evaluation/experiment.py`.

```
run_experiment.py       runs everything
src/rasff/              the pipeline
app/                    the dashboard
notebooks/              Colab version, imports the package
scripts/                one-off analyses
```

The test set is touched on exactly one line in `fit_score()` and nowhere else
in the project.

## Notes

The raw export isn't included. It's the European Commission's, not mine, so
download it yourself from the RASFF Window portal.

`results/` isn't committed either, with three exceptions: the trained model,
its card, and the 2,300-row held-out window the dashboard scores against.
Those are there so the app runs on clone. Everything else in `results/` is
generated, and a stale number sitting in a repo is how you end up quoting the
wrong figure.

The committed test window is a 2,300-row extract of the RASFF export, © European
Union, 2022-2025. Reused under Commission Decision 2011/833/EU. The Commission
is not responsible for anything I've done with it here. Source:
https://webgate.ec.europa.eu/rasff-window/

Seed 42 everywhere, so runs are repeatable.

The model is a random forest throughout. LightGBM is supported as an optional
comparison but isn't installed by default and isn't what ships — the argument
here is that a cheap portable model is good enough, so shipping a heavier one
would undercut it. `run_experiment.py` prints which models it found at startup.
