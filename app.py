"""LC Elo Analyzer — upload a private conversation export and explore reports.

Run locally:
    python -m streamlit run app.py

For hosted Streamlit deployments: browser uploads reach the hosting server. This app
keeps exports in memory and does not intentionally save, cache, or log their contents.
"""
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import io
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis_core import (
    DEFAULT_SCORE_WEIGHTS, analyze, generate_pdf, load_records_from_text,
    parse_common_words_csv, relabel_records, word_token,
)

st.set_page_config(page_title="LC Elo Analyzer", page_icon="📊", layout="wide")

# Matplotlib is not fully thread-safe; serialize PDF/chart rendering on the server.
_RENDER_LOCK = threading.RLock()
_PALETTE = [
    "#E15759", "#F28E2B", "#4E79A7", "#76B7B2", "#59A14F",
    "#B07AA1", "#9C755F", "#EDC948", "#FF9DA7", "#79706E",
]
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "*", "+"]


def _plot_hourly(records, aliases, styles):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    senders = sorted({r["sender"] for r in records})
    for sender in senders:
        counts = Counter(r["hour"] for r in records if r["sender"] == sender)
        total = sum(counts.values())
        y = [100.0 * counts.get(h, 0) / total if total else 0 for h in range(24)]
        style = styles.get(sender, {"color": "#888888", "marker": "o"})
        ax.plot(range(24), y, label=sender, color=style["color"],
                marker=style["marker"], linewidth=1.8, markersize=4)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=60, fontsize=8)
    ax.set(xlabel="Hour starting (24-hour clock)", ylabel="Share of sender's messages (%)",
           title="When members send messages")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=min(4, len(senders)))
    fig.tight_layout()
    return fig


def _plot_bars(df, column, title, ylabel, styles, label_format):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = df[column].tolist()
    bars = ax.bar(df["Member"], values,
                  color=[styles[s]["color"] for s in df["Member"]])
    ax.set(ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.margins(y=.16)
    for bar, value in zip(bars, values):
        ax.annotate(label_format.format(value),
                    (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_composition(message_counts, senders):
    from analysis_core import MESSAGE_COLORS
    fig, ax = plt.subplots(figsize=(11, 4.8))
    types = sorted({k for sender in senders for k in message_counts[sender]["Total"]})
    bottom = np.zeros(len(senders))
    for typ in types:
        vals = np.asarray([message_counts[s]["Total"].get(typ, 0) for s in senders])
        ax.bar(senders, vals, bottom=bottom, label=typ,
               color=MESSAGE_COLORS.get(typ, "#777777"))
        bottom += vals
    ax.set(ylabel="Text and media components", title="Message-type composition")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def _plot_daily(records):
    counts = Counter(r["date"] for r in records)
    date_index = pd.date_range(min(r["date"] for r in records),
                               max(r["date"] for r in records), freq="D")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(date_index, [counts.get(d.strftime("%Y-%m-%d"), 0) for d in date_index],
            marker="o", markersize=3, linewidth=1.7)
    ax.set(xlabel="Date", ylabel="Unique messages", title="Daily group activity")
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _summary_table(records, aliases, weights, word_denominator):
    rows = []
    by_sender = {}
    for r in relabel_records(records, aliases):
        by_sender.setdefault(r["sender"], []).append(r)
    for sender, member_records in sorted(by_sender.items()):
        types = Counter(t for r in member_records for t, _ in r["components"])
        words = sum(sum(bool(word_token(w)) for w in body.split())
                    for r in member_records for typ, body in r["components"] if typ == "Text")
        denominator = (len(member_records) if word_denominator == "all" else
                       sum(any(typ == "Text" for typ, body in r["components"])
                           for r in member_records))
        rows.append({
            "Member": sender,
            "Messages": len(member_records),
            "Components": sum(types.values()),
            "Total words": words,
            "Words/message": round(words / denominator, 2) if denominator else 0,
            "Score": round(sum(types[t] * weights.get(t, 1) for t in types), 2),
        })
    return pd.DataFrame(rows).sort_values("Messages", ascending=False)


st.title("LC Elo Analyzer")
st.caption("Open-source conversation analytics: customize participants, explore activity, and download a PDF report.")

with st.sidebar:
    st.header("1 · Conversation")
    uploads = st.file_uploader("Upload one or more conversation exports", type=["txt"],
                               accept_multiple_files=True,
                               help="Files are analyzed in memory. On a hosted site, they are transmitted to that server.")
    use_demo = st.checkbox("Use synthetic sample data", value=False,
                           disabled=bool(uploads))
    selected_bytes = None
    selected_name = None
    if uploads:
        options = list(range(len(uploads)))
        selected = st.selectbox("Select export to analyze", options,
                                format_func=lambda i: uploads[i].name)
        selected_bytes = uploads[selected].getvalue()
        selected_name = uploads[selected].name
    elif use_demo:
        selected_name = "sample_export.txt"
        selected_bytes = (Path(__file__).resolve().parent / selected_name).read_bytes()

    st.header("2 · Common-word filter")
    st.caption("Only used for the uncommon-word chart; total word counts remain unfiltered.")
    custom_csv = st.file_uploader("Optional common-words CSV", type=["csv"], key="common_csv")
    default_csv = Path(__file__).resolve().parent / "top_100_english_words.csv"
    common_words = set()
    try:
        if custom_csv:
            common_words = parse_common_words_csv(custom_csv.getvalue())
            st.success(f"Loaded {len(common_words)} common words from uploaded CSV")
        elif default_csv.exists():
            common_words = parse_common_words_csv(default_csv.read_bytes())
            st.success(f"Loaded {len(common_words)} common words from repository CSV")
        else:
            st.caption("No CSV loaded; uncommon-word chart includes all words.")
    except Exception as exc:
        st.error(f"Could not read common-word CSV: {exc}")
        st.stop()

if selected_bytes is None:
    st.info("Upload a `.txt` conversation export in the sidebar, or try the synthetic sample."
            " Your real conversation exports should never be committed to a public repository.")
    st.markdown("**What the app provides:** editable nicknames and colors, date selection, "
                "participant activity, word counts, reply analysis, and a downloadable multi-page PDF.")
    st.stop()

try:
    raw_text = selected_bytes.decode("utf-8-sig", errors="replace")
    all_records, skipped, n_blocks = load_records_from_text(raw_text)
except Exception as exc:
    st.error(f"Could not parse export: {exc}")
    st.stop()

if not all_records:
    st.error("No message records recognized. Confirm the file uses the supported conversation-export format.")
    st.stop()

raw_names = sorted({r["sender"] for r in all_records}, key=lambda x: (x != "You", x.lower()))
first_date = min(r["timestamp"].date() for r in all_records)
last_date = max(r["timestamp"].date() for r in all_records)
input_fingerprint = sha256(selected_bytes).hexdigest()[:16]

st.subheader("Participants")
st.write("Names are detected automatically from the export. Change how each person appears in charts and reports.")
cols = st.columns(2)
name_map = {}
styles = {}
for i, raw_name in enumerate(raw_names):
    with cols[i % 2]:
        with st.container(border=True):
            st.caption(f"Export name: {raw_name if raw_name != 'You' else 'You (sent from this account)'}")
            alias = st.text_input("Display name", value="Me" if raw_name == "You" else raw_name,
                                  key=f"alias-{input_fingerprint}-{raw_name}")
            color = st.color_picker("Chart color", value=_PALETTE[i % len(_PALETTE)],
                                    key=f"color-{input_fingerprint}-{raw_name}")
            name_map[raw_name] = alias.strip()
            styles[alias.strip()] = {"color": color, "marker": _MARKERS[i % len(_MARKERS)]}

if any(not nick for nick in name_map.values()):
    st.error("Every participant needs a display name.")
    st.stop()
if len(set(name_map.values())) != len(name_map):
    st.error("Display names must be unique; please resolve the duplicates above.")
    st.stop()

st.subheader("Analysis settings")
left, middle, right = st.columns(3)
with left:
    chosen_dates = st.date_input("Date range", value=(first_date, last_date),
                                 min_value=first_date, max_value=last_date)
with middle:
    selected_names = st.multiselect("Participants", options=raw_names, default=raw_names,
                                    format_func=lambda n: name_map[n])
with right:
    explicit_only = st.checkbox("Only count explicit replies", value=False,
                                help="When unchecked, a sender change can be treated as an inferred reply.")
    word_denominator = st.radio("Words per message denominator", ["all", "text"],
                                format_func=lambda v: "All unique messages" if v == "all"
                                                       else "Messages with text only")

with st.expander("Customize scoring weights", expanded=False):
    weight_cols = st.columns(4)
    weights = {}
    for i, (category, initial) in enumerate(DEFAULT_SCORE_WEIGHTS.items()):
        with weight_cols[i % len(weight_cols)]:
            weights[category] = st.number_input(category, value=float(initial),
                                                step=0.5, key=f"weight-{category}")

if not isinstance(chosen_dates, (tuple, list)) or len(chosen_dates) != 2:
    st.info("Choose both a start date and an end date.")
    st.stop()
start_date, end_date = chosen_dates
if start_date > end_date:
    st.error("Start date must come before end date.")
    st.stop()
if not selected_names:
    st.info("Select at least one participant.")
    st.stop()

records = [r for r in all_records if r["sender"] in selected_names
           and start_date <= r["timestamp"].date() <= end_date]
if not records:
    st.warning("No messages match the chosen dates and participants.")
    st.stop()

mapped_records = relabel_records(records, name_map)
message_counts, hourly, replies, texts, explicit_n, inferred_n, unmatched = analyze(
    mapped_records, name_map, only_explicit_replies=explicit_only
)
summary = _summary_table(records, name_map, weights, word_denominator)

st.divider()
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique messages", f"{len(records):,}")
k2.metric("Message components", f"{sum(len(r['components']) for r in records):,}")
k3.metric("Participants", len(selected_names))
k4.metric("Explicit / inferred replies", f"{explicit_n:,} / {inferred_n:,}")
st.dataframe(summary, hide_index=True, use_container_width=True)
st.download_button("Download summary CSV", data=summary.to_csv(index=False).encode("utf-8-sig"),
                   file_name="lc_elo_summary.csv", mime="text/csv")

tabs = st.tabs(["Activity", "Message composition", "Words", "Replies", "PDF report"])
with tabs[0]:
    with _RENDER_LOCK:
        fig = _plot_daily(mapped_records)
        st.pyplot(fig)
        plt.close(fig)
        fig = _plot_hourly(mapped_records, name_map, styles)
        st.pyplot(fig)
        plt.close(fig)
with tabs[1]:
    with _RENDER_LOCK:
        fig = _plot_composition(message_counts, list(summary["Member"]))
        st.pyplot(fig)
        plt.close(fig)
with tabs[2]:
    a, b = st.columns(2)
    with _RENDER_LOCK:
        fig = _plot_bars(summary, "Total words", "Total words (not filtered)",
                         "Words", styles, "{:,.0f}")
        with a:
            st.pyplot(fig)
        plt.close(fig)
        fig = _plot_bars(summary, "Words/message", "Average words per message",
                         "Words per message", styles, "{:.2f}")
        with b:
            st.pyplot(fig)
        plt.close(fig)
    st.caption("Total word counts include common words. Consecutive emojis separated by no spaces count as one token, matching the original script.")
with tabs[3]:
    data = []
    for sender in sorted(replies):
        for receiver, number in sorted(replies[sender].items()):
            if number > 0:
                data.append({"Sender": sender, "Receiver": receiver, "Replies": number})
    if data:
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    else:
        st.info("No replies identified with these settings.")
    st.caption("If inference is enabled, a sender change is treated as a possible reply, not a confirmed one.")
with tabs[4]:
    st.write("Generate the complete original-style PDF with your selected dates, participants, nicknames, colors, and scoring weights.")
    config_signature = sha256(repr((input_fingerprint, chosen_dates, selected_names,
                                    name_map, styles, weights, explicit_only,
                                    word_denominator, sorted(common_words))).encode()).hexdigest()
    if st.button("Generate PDF report", type="primary"):
        try:
            with st.spinner("Creating report..."):
                with _RENDER_LOCK:
                    pdf_bytes, report_info = generate_pdf(
                        records, skipped, n_blocks, name_map, styles,
                        common_words=common_words, score_weights=weights,
                        only_explicit_replies=explicit_only,
                        words_denominator=word_denominator
                    )
            st.session_state["generated_pdf"] = (config_signature, pdf_bytes, report_info)
        except Exception as exc:
            st.error(f"Could not generate PDF: {exc}")
    pdf_state = st.session_state.get("generated_pdf")
    if pdf_state and pdf_state[0] == config_signature:
        _, pdf_bytes, info = pdf_state
        filename = f"LC_Elo_{Path(selected_name).stem}_{start_date}_{end_date}.pdf"
        st.download_button("Download PDF report", data=pdf_bytes, file_name=filename,
                           mime="application/pdf", type="primary")
        st.caption(f"{info['unique_messages']:,} unique messages included. Nothing was written to disk by this application.")
    elif pdf_state:
        st.info("Settings changed. Regenerate the report to apply the new selections.")

st.caption("Privacy: source code can be public without publishing conversation exports. On Streamlit Community Cloud, uploaded content is sent to the hosting server; run locally if it must stay on your device.")
