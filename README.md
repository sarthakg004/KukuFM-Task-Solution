# Kuku Promo Finder

Find and rank 20–60-second promos from timestamped microdrama screenplays.
Each result includes the episode, ordered segment timestamps, hook, cliffhanger,
explanation and posting rationale.

[Solution document](docs/solution.pdf) · [Designed HTML version](docs/solution.html)

## Workflow

**Parser → Candidate generation → Deduplication → Ranking**

1. **Parser:** reads the original text, preserves timestamps and assigns source IDs.
2. **Candidate generation:** Gemini proposes clips across overlapping episode windows.
   Code resolves their IDs to timestamps and checks duration and boundaries.
3. **Deduplication:** rules group matching footage by episode, timestamps and segment
   order. All alternative cuts are retained.
4. **Ranking:** a separate judge scores the actual excerpts, chooses the strongest
   cut in each group, and compares the leading shortlist for posting order.

The tool selects footage and cut points. An editor reviews the final shortlist.

## Directory structure

```text
solution.ipynb             Configuration and complete workflow
scripts/                   Python implementation
  preprocessing.py         Parser and episode windows
  promo_generator.py       Gemini calls, caching and candidate validation
  api_keys.py              Automatic discovery of configured Gemini keys
  request_budget.py        Persistent daily request budgets
  deduplication.py          Timestamp-based duplicate rules
  promo_judge.py           Evidence-based scores and ranking
prompts/                   All model prompts
  candidate_prompt.txt
  judge_prompt.txt
  posting_prompt.txt
tests/                     Automated checks with synthetic fixtures
docs/                      Solution PDF and HTML
outputs/                   Local results; excluded from Git
```

## Setup and run

Use Python 3.11. From the repository folder:

```bash
conda create -n kukufm python=3.11
conda activate kukufm
python -m pip install -r requirements.txt jupyterlab ipykernel
cp .env.example .env
python -m jupyter lab solution.ipynb
```

Skip environment creation if `kukufm` already exists. Add at least one available
Gemini key to `.env`, select the `kukufm` Python environment in Jupyter, and set
`INPUT_DIR` to the authorized screenplay folder. Run the notebook from top to bottom.
The input pack is not included in this repository.

Keys are detected automatically: use `GEMINI_API_KEY` or numbered variables such
as `GEMINI_API_KEY1`, `GEMINI_API_KEY2`, etc. Any count is supported, numbering may
have gaps, and blank or duplicate values are skipped. No key list is needed in the
notebook. Keys from `.env` and the environment share the existing budget and rotation.


## Configuration

The first notebook cell contains all hyperparameters: paths, model list, episodes
per window, stride, candidate ceiling, thinking level, token budget, retries, delays,
daily request cap, deduplication threshold and judge weights.

The completed run used **Gemini 3.7, 10 episodes per window, stride 5, up to 8
candidates and medium thinking**. Ranking weights are hook 30%, cliffhanger 30%,
clarity 25% and escalation 15%; these are editable assumptions.

Successful API responses are cached. Rerun to resume after interruption. To rank
saved candidates directly, set `JUDGE_INPUT_PATH` and run configuration,
preprocessing and Section 6. Changing prompts or model settings can require new calls.

## Results

The full 50-episode pack produced **51 technically valid candidates**, grouped into
**35 promo groups** with 16 alternative cuts retained. The judge ranked all 35
groups, flagged 8 below the configured core-quality threshold, and selected a
stronger alternative cut in 2 groups.

| Folder | Contents |
|---|---|
| `outputs/candidates/` | Raw model responses, attempts and validated candidates |
| `outputs/deduplicated/` | Representatives and all alternative cuts |
| `outputs/ranking/` | Raw judge responses and ranked JSON/Markdown reports |

The final ranking includes every segment's episode and start/end timestamps,
hook, cliffhanger, explanation, scores, evidence and posting rationale.

## Tests and limitations

```bash
python -m unittest discover -s tests -v
```

Tests mock API calls. Private-pack integration tests are skipped when the pack is
absent; synthetic tests still run. The judge reads screenplay descriptions rather
than finished video and may share the generator's biases. Word-level cut accuracy
is uncertain. Differently assembled montages can escape deduplication. Scores are
editorial judgments, not measured advertising performance.

Daily budgets persist across restarts. The supplied keys use separate projects;
keys within one Google Cloud project share quota. The local ledger cannot see
requests made earlier or outside this pipeline. API errors may still interrupt a run.

## Solution document and AI use

The finished two-page [solution PDF](docs/solution.pdf) and
[HTML version](docs/solution.html) explain the pipeline, show an illustrative
example, and document results and limitations. PDF-generation utilities are not
part of this repository.

Codex assisted planning, implementation, debugging and documentation. Gemini proposed
and judged promos. An early output-budget setting caused truncated JSON; finish-status
checks now reject incomplete responses.

The screenplay pack, `.env` and outputs are excluded from Git. Notebook outputs are
cleared before sharing. Confidential outputs are provided separately to the reviewers.
