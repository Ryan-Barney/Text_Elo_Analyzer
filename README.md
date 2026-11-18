# LC Elo Analyzer — Open-source web starter

Turn a group-chat `.txt` export into a configurable dashboard and downloadable PDF.
This is a **Streamlit application** adapted from a text analysis script.
It outputs report charts and calculations while removing hard-coded
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
