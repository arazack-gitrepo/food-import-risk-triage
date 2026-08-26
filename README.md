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
`data/RASFF_window.csv`. Download from 2022 even though the model only trains
on 2023 onward, because the 2022 rows are what show the labelling scheme
changed.

That one command regenerates everything in `results/`. Takes about 10 to 15
minutes on a normal laptop. `--quick` runs a shorter version if you just want
to see it work.

To open the dashboard afterwards:

```bash
streamlit run app/main.py
```

If that says streamlit isn't recognised, use `python -m streamlit run app/main.py`.
The dashboard works straight after cloning, you don't need the data for it.

## What I found

The model scores 0.839 macro F1 using the full RASFF record. But most of that
comes from fields the inspector doesn't have yet when they're deciding what to
open:

| what the model knows | macro F1 |
|---|---|
| just the customs declaration | 0.686 |
| + notifying country | 0.699 |
| + notification type (a judgement already made by the filing official) | 0.810 |
| + hazard category (usually a lab result) | 0.839 |

So 0.839 is what's achievable with the complete record, and 0.686 is what's
achievable before anything is opened. Both are worth reporting. Quoting only
the first overstates what the model actually knows at decision time.

The biggest single jump is notification type, worth +0.111. Each step is
compared against the one directly below it, not against the bottom, so that
number is a tested difference rather than two scores subtracted.

The written description is worth more than I first thought. Against the full
record it looks useless, but against what's actually known before inspection
it adds +0.088. The earlier null result came from comparing it against a
baseline that already had the lab result in it.
`scripts/reconcile_baseline.py` runs both comparisons side by side.

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
`src/rasff/evaluation/experiment.py`. The test set is touched on exactly one
line in there and nowhere else in the project.

```
run_experiment.py       runs everything
src/rasff/              the pipeline
app/                    the dashboard
notebooks/              Colab version, imports the package
scripts/                one-off analyses
```

## Notes

The data isn't included. It's the European Commission's, not mine, you can download
it yourself from the RASFF Window portal. The one exception is the 2,300-row
test window in `results/predictions/`, which is there so the dashboard runs on
clone. That's European Union, 2022-2025

Nothing else in `results/` is committed apart from the trained model and its
card. Everything in there is generated, and a stale number sitting in a repo
is how you end up quoting the wrong figure.

Seed 42 everywhere, so runs are repeatable.

The model is a random forest. LightGBM is optional and only used for
comparison. the whole argument here is that a cheap portable model is good
enough, so shipping a heavier one would undercut it.

Commits appear under two GitHub accounts (arazack-gitrepo and primenutron71). Both are mine, i did work done across two machines signed into different accounts.
