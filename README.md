# LC Elo Analyzer — Open-source web starter

Turn a private group-chat `.txt` export into a configurable dashboard and downloadable PDF.
This is an **initial working Streamlit application** adapted from the LC Elo analysis script.
It retains the existing report charts and calculations while removing hard-coded
participants, names, and colors. The repository contains **no real conversations**.

## Features

- Select an uploaded conversation export (or use a synthetic sample).
- Parse old and UUID-augmented exports; merge media + caption with shared UUID.
- Automatically detect participant names; customize display names and chart colors.
- Filter by date and participants.
- Explore daily activity, hourly proportions, word counts, message composition, and replies.
- Select whether inferred replies count; customize scoring weights and words/message denominator.
- Download a CSV summary and full multi-page PDF.
- Optionally add the public `top_100_english_words.csv` with a `word` column to the repo,
  or upload it through the interface.

## Run locally (recommended when files must stay on your computer)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in the browser at `http://localhost:8501`. Close the terminal or stop
Streamlit when finished. If Python 3.12 is not installed, choose an installed supported
version using `py -0p` (Windows). On macOS/Linux, use `python3 -m venv .venv`
and `.venv/bin/python -m streamlit run app.py`.

## Make source code public on GitHub

1. Create a public repository (for example, `lc-elo-analyzer`).
2. Add `app.py`, `analysis_core.py`, `requirements.txt`, `.gitignore`, this README,
   `sample_export.txt`, and `tests/`.
3. Optionally add `top_100_english_words.csv` (only the generic word list, no personal data; verify redistribution permission).
4. **Never add** real chat export files, PDF reports, personal configuration files,
   or data directories. The `.gitignore` provides guardrails but cannot undo a past commit.
5. Run `git status` and inspect all tracked files before pushing.

## Publish a hosted site (optional)

On [Streamlit Community Cloud](https://share.streamlit.io), create a new app from your
GitHub repository and select `app.py` as the main file. Streamlit handles installing
packages from `requirements.txt` and gives the app a public URL.

**Privacy warning:** Hosted Streamlit uploads are sent to a remote backend. This starter
does not intentionally save uploaded data to disk, place it in a global cache, or log
raw message bodies, but **that is not equivalent to browser-only processing**.
Do not promise end-to-end privacy on third-party hosting. If strict local-only
processing is required, run the Streamlit app locally or later port to a
client-side JavaScript/Pyodide application hosted on GitHub Pages.

## Files

```
app.py               Streamlit user interface
analysis_core.py     Parsing, counting, plotting, PDF creation
requirements.txt     Python dependencies
sample_export.txt    Synthetic, nonprivate example
.gitignore           Blocks common private files from accidental Git adds
tests/test_core.py   Parser, tokenization, and PDF tests
```

## Test locally

```powershell
python -m unittest discover -s tests -v
```

## Known limitations / next steps

- The exporter format is specific; new formats may require a parser adapter.
- Explicit reply targets whose names differ from participant names may not map
  automatically and are reported as unmatched.
- When selecting a subset of members, reply statistics reflect the subset and
  may exclude replies involving people outside it.
- Matplotlib is serialized during rendering; large reports can take longer.
- An advanced privacy-first GitHub Pages version would run analysis entirely in the browser,
  but cannot run this Python server script unchanged.
- An MIT `LICENSE` is supplied; review the copyright attribution and license choice before publishing.
