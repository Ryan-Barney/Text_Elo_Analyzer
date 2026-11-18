"""Pure-ish analysis backend for the LC Elo website.

The app passes uploaded text in memory and participant settings as arguments.
No private names or message exports are part of this source file.
"""
import io
import re
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns

MESSAGE_COLORS = {
    "Audio": "orange", "Image": "red", "Link": "green", "Reaction": "gold",
    "Text": "blue", "Unsend": "gray", "Video": "purple", "Attachment": "brown",
}
DEFAULT_SCORE_WEIGHTS = {
    "Unsend": -10.0, "Image": 2.0, "Audio": 2.0, "Video": 4.0,
    "Attachment": 4.0, "Link": 3.0, "Reaction": 0.5, "Text": 1.0,
}

# Divider must appear on its own line; message text may have other hyphens.
DIVIDER_RE = re.compile(r"(?m)^-{20,}[ \t]*$")
TIMESTAMP_RE = re.compile(
    r"^(?P<datetime>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?P<rest>.*)$"
)
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$")
REPLY_RE = re.compile(
    r"^↩\s*Replying to\s+(.+?),\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}:"
)
REACTION_PREFIXES = (
    "Loved", "Liked", "Disliked", "Laughed at", "Emphasized", "Questioned",
    "Le encantó", "Le gustó", "No le gustó", "Le dio risa", "Exclamó por", "Cuestionó",
)
REACTION_DETAILS = (
    "“", '"', "an image", "a movie", "an audio message", "Digital Touch Message",
)


def categorize_message(content):
    """Return a single message category, looking at the *actual* body only."""
    content = content.strip()
    if re.fullmatch(r".+? unsent a message\.", content, flags=re.DOTALL):
        return "Unsend"
    if re.fullmatch(r".+? (?:added .+? to|removed .+? from) the conversation\.", content):
        return "AddRemove"
    if content.endswith(" kept an audio message from you."):
        return "Keepers"
    if re.fullmatch(r".+? (?:changed|removed) the background\.", content):
        return "Notification"
    if content.startswith("Reacted with a sticker to") or re.match(r"^Reacted\s+.+?\s+to\b", content):
        return "Reaction"
    if content.startswith("Reaccionó con"):
        return "Reaction"
    if any(content.startswith(prefix + " " + detail)
           for prefix in REACTION_PREFIXES for detail in REACTION_DETAILS):
        return "Reaction"
    # These are labels of the media *sent*, not references to a prior message.
    if content.endswith("(Image)"):
        return "Image"
    if content.endswith("(Video)"):
        return "Video"
    if content.endswith("(Audio)"):
        return "Audio"
    if content.endswith("(Attachment)"):
        return "Attachment"
    if "https://" in content or "http://" in content:
        return "Link"
    return "Text"


def parse_message_block(block):
    """Parse one old/new block; do not depend on fixed line positions.

    A current block is: chat title, UUID, timestamp, blank, body.
    An older block is: chat title, timestamp, blank, body.
    Explicit replies have an extra quote/header before the actual body.
    """
    lines = block.strip().splitlines()
    found = None
    # Restrict search to metadata lines so dates inside the body aren't mistaken
    # for a message timestamp.
    for i, line in enumerate(lines[:6]):
        m = TIMESTAMP_RE.match(line.strip())
        if m:
            found = (i, m)
            break
    if found is None:
        return None, "No timestamp line"

    timestamp_index, match = found
    try:
        dt = datetime.strptime(match.group("datetime"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None, "Invalid timestamp"

    body = "\n".join(lines[timestamp_index + 1:]).strip()
    if not body:
        return None, "Empty message"

    # The new exporter writes explicit reply metadata separately from the reply.
    explicit_target = None
    reply_match = REPLY_RE.match(body)
    if reply_match:
        explicit_target = reply_match.group(1).strip()
        # Original quoted message and new response are separated by a blank line.
        pieces = re.split(r"\n\s*\n", body, maxsplit=1)
        if len(pieces) != 2 or not pieces[1].strip():
            return None, "Reply header without a message"
        body = pieces[1].strip()

    category = categorize_message(body)
    rest = match.group("rest").strip()

    # System notifications don't identify the sender on the timestamp line.
    if rest == "notification":
        if category != "Unsend":
            return None, "System notification"
        sender_match = re.fullmatch(r"(.+?) unsent a message\.", body)
        if not sender_match:
            return None, "Unsent notification without sender"
        sender_name = sender_match.group(1)
        if sender_name.lower() == "you":
            sender_name = "You"
    else:
        sender_match = re.search(r"\bfrom\s+(.+?)\s+\(", rest)
        if sender_match:
            sender_name = sender_match.group(1).strip()
        elif rest.startswith("to "):
            sender_name = "You"
        else:
            return None, "Unrecognized sender metadata"

    sender = sender_name

    message_id = None
    for line in lines[:timestamp_index]:
        if UUID_RE.fullmatch(line.strip()):
            message_id = line.strip().lower()
            break

    return {
        "timestamp": dt,
        "date": dt.strftime("%Y-%m-%d"),
        "hour": dt.hour,
        "sender": sender,
        "body": body,
        "category": category,
        "explicit_target": explicit_target,
        "message_id": message_id,
    }, None


def nickname(name, name_map):
    if name is None:
        return None
    raw = name.strip()
    if raw.lower() in {"you", "me"}:
        raw = "You"
    return name_map.get(raw) or (raw if raw in name_map.values() else None)


def load_records_from_text(data):
    """Return *logical messages* grouped by UUID, retaining all components.

    The new export can repeat a UUID for image(s) + accompanying caption.
    Every image/text component is included in composition/word counts, but the
    underlying message contributes only once to daily/hourly and reply counts.
    Older exports do not have UUIDs; each block counts as one message.
    """
    data = data.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b for b in DIVIDER_RE.split(data) if b.strip()]
    skipped = Counter()
    grouped = {}
    for i, block in enumerate(blocks):
        record, reason = parse_message_block(block)
        if record is None:
            skipped[reason] += 1
            continue
        msg_key = ("uuid", record["message_id"]) if record["message_id"] else ("block", i)
        if msg_key not in grouped:
            record["components"] = [(record["category"], record["body"])]
            grouped[msg_key] = record
        else:
            existing = grouped[msg_key]
            if (existing["sender"] != record["sender"] or
                    existing["timestamp"] != record["timestamp"]):
                # Bad or reused UUID; do not merge different senders/timepoints.
                record["components"] = [(record["category"], record["body"])]
                grouped[("collision", i)] = record
                skipped["UUID collision (kept separately)"] += 1
            else:
                existing["components"].append((record["category"], record["body"]))
                if existing["explicit_target"] is None:
                    existing["explicit_target"] = record["explicit_target"]

    records = list(grouped.values())
    records.sort(key=lambda x: x["timestamp"])
    return records, skipped, len(blocks)


def format_text_pages(pdf, text, title, max_lines_per_page=55, line_width=105):
    wrapped = []
    for line in text.splitlines():
        wrapped.extend(textwrap.wrap(line, width=line_width, replace_whitespace=False,
                                     drop_whitespace=False) or [""])
    for start in range(0, max(1, len(wrapped)), max_lines_per_page):
        part = wrapped[start:start + max_lines_per_page]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.99, title, transform=ax.transAxes, va="top",
                fontsize=13, fontweight="bold")
        ax.text(0.02, 0.95, "\n".join(part), transform=ax.transAxes,
                family="DejaVu Sans", fontsize=8, va="top")
        pdf.savefig(fig)
        plt.close(fig)


def add_pie(pdf, labels, values, colors, title):
    positive = [(label, val) for label, val in zip(labels, values) if val > 0]
    if not positive:
        return False
    labels, values = zip(*positive)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=labels, colors=[colors.get(x, "gray") for x in labels],
           autopct="%1.1f%%", textprops={"fontsize": 9})
    ax.set_title(title)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_common_words_csv(csv_content):
    """Load a public common-words CSV, without retaining the uploaded file."""
    if isinstance(csv_content, bytes):
        csv_content = csv_content.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(csv_content))
    if "word" not in df.columns:
        raise ValueError("Common words CSV must contain a column called 'word'.")
    return {str(x).strip().replace("’", "'").upper()
            for x in df["word"].dropna()}


def word_token(word):
    """Keep adjacent emojis together, as in the original whitespace tokenizer."""
    return (word.upper().replace(".", "").replace("!", "")
            .replace("?", "").replace(",", "").replace('"', "")
            .replace("’", "'"))


def analyze(records, name_map, only_explicit_replies=False):
    message_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    hourly_counts = defaultdict(lambda: defaultdict(int))
    reply_counts = defaultdict(lambda: defaultdict(int))
    processed_texts = []
    previous_sender = None
    explicit_replies = 0
    inferred_replies = 0
    unmatched_reply_targets = Counter()

    for record in records:
        sender, date = record["sender"], record["date"]
        # One logical message may contain multiple images and/or text captions.
        # Retain each component in the category and scoring tallies.
        for typ, body in record["components"]:
            message_counts[sender][date][typ] += 1
            message_counts[sender]["Total"][typ] += 1
            if typ == "Text":
                processed_texts.append(body)
        # But a single sent message contributes just once to the hourly curve.
        hourly_counts[sender][record["hour"]] += 1

        if record["explicit_target"] is not None:
            receiver = nickname(record["explicit_target"], name_map)
            if receiver is None:
                unmatched_reply_targets[record["explicit_target"]] += 1
            elif receiver != sender:
                explicit_replies += 1
                reply_counts[sender][receiver] += 1
                message_counts[sender]["Replying_to_" + receiver][record["category"]] += 1
        elif not only_explicit_replies and previous_sender is not None and previous_sender != sender:
            inferred_replies += 1
            reply_counts[sender][previous_sender] += 1
            message_counts[sender]["Replying_to_" + previous_sender][record["category"]] += 1
        previous_sender = sender

    return (message_counts, hourly_counts, reply_counts, processed_texts,
            explicit_replies, inferred_replies, unmatched_reply_targets)




def relabel_records(records, name_map):
    """Return renamed records without modifying the parsed originals."""
    return [{**record, "sender": name_map.get(record["sender"], record["sender"])}
            for record in records]


def generate_pdf(raw_records, skipped, n_blocks, name_map, sender_styles,
                 common_words=None, score_weights=None, only_explicit_replies=False,
                 words_denominator="all"):
    """Return (PDF bytes, report summary). Never writes the private export to disk."""
    if not raw_records:
        raise ValueError("No valid messages found in the export.")
    if words_denominator not in {"all", "text"}:
        raise ValueError("words_denominator must be 'all' or 'text'.")
    records = relabel_records(raw_records, name_map)
    score_weights = score_weights or DEFAULT_SCORE_WEIGHTS
    (message_counts, hourly_counts, reply_counts, processed_texts,
     explicit_replies, inferred_replies, unmatched_targets) = analyze(
        records, name_map, only_explicit_replies=only_explicit_replies
    )
    senders = list(message_counts)
    component_count = sum(len(r["components"]) for r in records)
    first_day = records[0]["timestamp"].date()
    last_day = records[-1]["timestamp"].date()
    date_objects = [first_day + timedelta(days=i)
                    for i in range((last_day - first_day).days + 1)]
    dates = [d.isoformat() for d in date_objects]
    earliest_timestamp, latest_timestamp = dates[0], dates[-1]
    pie_colors = {name: style["color"] for name, style in sender_styles.items()}
    common_words = common_words or set()
    lc_words = Counter()
    for content in processed_texts:
        for raw_word in content.split():
            word = word_token(raw_word)
            if word and word not in common_words:
                lc_words[word] += 1
    pdf_buffer = io.BytesIO()
    # New: count ALL words in each member's Text components. Unlike lc_words,
    # these counts do NOT exclude entries in the common-words CSV.
    sender_word_totals = Counter()
    for record in records:
        for typ, body in record["components"]:
            if typ == "Text":
                sender_word_totals[record["sender"]] += sum(
                    bool(word_token(raw_word)) for raw_word in body.split()
                )

    # Each UUID-grouped record counts once, even if it contains media + caption.
    sender_message_totals = Counter(
        record["sender"] for record in records
        if words_denominator == "all"
        or any(typ == "Text" for typ, body in record["components"])
    )
    sender_avg_words = {
        sender: (sender_word_totals[sender] / sender_message_totals[sender]
                 if sender_message_totals[sender] else 0.0)
        for sender in senders
    }

    status_lines = [f"Calculating status for LC from {earliest_timestamp} to {latest_timestamp}:", ""]
    for sender in senders:
        counts = message_counts[sender]["Total"]
        score = sum(count * score_weights.get(typ, 1) for typ, count in counts.items())
        if score >= 1000:
            status = "Honorable LC member"
        elif score >= 600:
            status = "Active LC member"
        elif score >= 300:
            status = "Stagnant LC member"
        else:
            status = "Fall Off"
        status_lines.extend([f"Member ID: {sender}", f"Score: {score:g} | Status: {status}", ""])
    status_lines.extend([
        f"Export blocks inspected: {n_blocks}",
        f"Unique messages counted: {len(records)}",
        f"Message components counted by type: {component_count}",
        f"Explicit replies: {explicit_replies}",
        f"Inferred sender-change: {inferred_replies}",
        # f"Common-word filter: {common_path if common_path else 'NOT FOUND (no words excluded)'}",
        "",
        "Skipped entries:",
    ])
    status_lines.extend(f"  {reason}: {count}" for reason, count in skipped.most_common())
    if unmatched_targets:
        status_lines.extend(["", "Unrecognized explicit reply targets:"])
        status_lines.extend(f"  {name}: {count}" for name, count in unmatched_targets.most_common())

    with PdfPages(pdf_buffer) as pdf:
        format_text_pages(pdf, "\n".join(status_lines), "LC Activity Summary")

        # 1) Cumulative message count by day, including any empty intervening days.
        counts_by_day = Counter(r["date"] for r in records)
        cumulative = np.cumsum([counts_by_day[d] for d in dates])
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(dates, cumulative, marker="*", markersize=8, linewidth=2)
        if len(dates) <= 30:
            for x, y in zip(dates, cumulative):
                ax.annotate(str(y), (x, y), xytext=(0, 6), textcoords="offset points",
                            fontsize=8, ha="center")
        ax.set(xlabel="Date", ylabel="Cumulative message count",
               title=f"Total LC Messages from {earliest_timestamp} to {latest_timestamp}")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        if len(dates) > 35:
            ax.set_xticks(dates[::max(1, len(dates) // 25)])
        ax.grid(True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2) Score by day, with scoring consistent with the first summary page.
        fig, ax = plt.subplots(figsize=(12, 8))
        for sender in senders:
            style = sender_styles.get(sender, {"color": "gray", "marker": "o"})
            day_scores = [sum(count * score_weights.get(typ, 1)
                              for typ, count in message_counts[sender].get(d, {}).items())
                          for d in dates]
            ax.plot(dates, day_scores, label=sender, markersize=6,
                    linewidth=2, **style)
        ax.set(xlabel="Date", ylabel="Daily score",
               title=f"LC Member Scores Each Day ({earliest_timestamp} to {latest_timestamp})")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        if len(dates) > 35:
            ax.set_xticks(dates[::max(1, len(dates) // 25)])
        ax.grid(True)
        ax.legend(title="LC Members", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3) Percent of each member's own messages falling in each hour bin.
        fig, ax = plt.subplots(figsize=(12, 8))
        for sender in senders:
            style = sender_styles.get(sender, {"color": "gray", "marker": "o"})
            hourly_values = np.array([hourly_counts[sender].get(h, 0) for h in range(24)])
            total = hourly_values.sum()
            percentages = hourly_values * 100 / total if total else hourly_values
            ax.plot(range(24), percentages, label=sender, markersize=6,
                    linewidth=2, **style)
        hour_labels = [f"{h}:00-{(h + 1) % 24}:00" for h in range(24)]
        ax.set_xticks(range(24), hour_labels, rotation=50, fontsize=7)
        ax.set(xlabel="Hour of day", ylabel="Percentage of sender's messages (%)",
               title="LC Member Activity Over 24 Hours")
        ax.grid(True)
        ax.legend(title="LC Members", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4) Stacked composition (text/media *components* by sender and type).
        message_types = sorted({k for s in senders for k in message_counts[s]["Total"]},
                               key=lambda typ: -sum(message_counts[s]["Total"].get(typ, 0)
                                                    for s in senders))
        fig, ax = plt.subplots(figsize=(10, 7))
        x = np.arange(len(senders))
        bottom = np.zeros(len(senders))
        for typ in message_types:
            scores = np.array([message_counts[s]["Total"].get(typ, 0) for s in senders])
            ax.bar(x, scores, bottom=bottom, label=typ,
                   color=MESSAGE_COLORS.get(typ, "lightgray"))
            for i, score in enumerate(scores):
                if score > 20:
                    ax.text(i, bottom[i] + score / 2, str(score), ha="center",
                            va="center", fontsize=8)
            bottom += scores
        ax.set_xticks(x, senders, rotation=45)
        ax.set(xlabel="Member IDs", ylabel="Total text/media components",
               title="Message-Type Components for Each LC Member")
        ax.legend(title="Message Types", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 5) All words written by each sender (NO common-word filtering).
        # Keep the same order and participant colors in both new plots.
        word_plot_senders = sorted(senders, key=lambda s: sender_word_totals[s], reverse=True)
        x_words = np.arange(len(word_plot_senders))
        word_colors = [sender_styles.get(s, {"color": "gray"})["color"]
                       for s in word_plot_senders]
        word_totals = [sender_word_totals[s] for s in word_plot_senders]

        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.bar(x_words, word_totals, color=word_colors)
        for bar, total in zip(bars, word_totals):
            ax.annotate(f"{total:,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x_words, word_plot_senders, rotation=35)
        ax.set(xlabel="LC member", ylabel="Total words",
               title=f"Total Words Sent by LC Members ({earliest_timestamp} to {latest_timestamp})")
        ax.margins(y=0.12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 6) Words per unique message, using the selected denominator.
        # UUID-grouped messages count once.
        average_words = [sender_avg_words[s] for s in word_plot_senders]
        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.bar(x_words, average_words, color=word_colors)
        for bar, average in zip(bars, average_words):
            ax.annotate(f"{average:.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x_words, word_plot_senders, rotation=35)
        ax.set(xlabel="LC member", ylabel="Average words per unique message",
               title=f"Average Words per Message ({earliest_timestamp} to {latest_timestamp})")
        ax.margins(y=0.12)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 7) Outgoing reply distribution: denominator is sender's total replies.
        # If fallback is enabled, a change of sender can count as an inferred reply.
        for sender in senders:
            targets = sorted(s for s in senders if s != sender)
            add_pie(pdf, targets, [reply_counts[sender].get(s, 0) for s in targets],
                    pie_colors, f"{sender}'s Reply Distribution")

        # 8) Incoming reply distribution: denominator is replies *to* that receiver.
        for receiver in senders:
            responders = sorted(s for s in senders if s != receiver)
            add_pie(pdf, responders, [reply_counts[s].get(receiver, 0) for s in responders],
                    pie_colors, f"{receiver}'s Receiving Distribution")

        # 9) Preserve the old normalized relationship-score formula, but avoid
        # mutating defaultdicts during reads. Each sender's nonzero row sums to 10.
        sender_replies_norm_scaled = {}
        outgoing_totals = {s: sum(reply_counts[s].values()) for s in senders}
        for sender in senders:
            normalized = {}
            for receiver in senders:
                if sender == receiver:
                    normalized[receiver] = 0
                else:
                    share = (reply_counts[sender].get(receiver, 0) / outgoing_totals[sender]
                             if outgoing_totals[sender] else 0)
                    denom = outgoing_totals[receiver]
                    normalized[receiver] = share / denom if denom else 0
            total = sum(normalized.values())
            sender_replies_norm_scaled[sender] = {
                receiver: (10 * norm / total if total else 0)
                for receiver, norm in normalized.items()
            }
        relation_df = pd.DataFrame.from_dict(sender_replies_norm_scaled, orient="index")
        relation_df = relation_df.reindex(index=senders, columns=senders).fillna(0)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(relation_df, annot=True, fmt=".2f", cmap="Greens",
                    cbar_kws={"label": "Normalized relationship score"}, ax=ax)
        ax.set(xlabel="Receiver", ylabel="Sender",
               title=f"LC Member Send-Receive Relationships ({earliest_timestamp} to {latest_timestamp})")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 10) Top uncommon words. Whitespace tokenization retains contiguous
        # emoji sequences as single 'words'; quoted reply text is excluded.
        if lc_words:
            top_words = lc_words.most_common(100)
            word_df = pd.DataFrame(top_words, columns=["Word", "Total"])
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.bar(np.arange(len(word_df)), word_df["Total"])
            ax.set_xticks(np.arange(len(word_df)), word_df["Word"],
                          fontsize=6, rotation=90)
            ax.set(xlabel="Words", ylabel="Frequency",
                   title=f"Top 100 Uncommon Words ({earliest_timestamp} to {latest_timestamp})")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 11) Full count breakdown on multiple letter-size pages.
        detailed_lines = []
        for sender in senders:
            detailed_lines.extend([f"Sender: {sender}"])
            for date in dates:
                counts = message_counts[sender].get(date)
                if not counts:
                    continue
                detailed_lines.extend([f"  Date: {date}"])
                detailed_lines.extend(f"    {typ}: {n}" for typ, n in sorted(counts.items()))
            detailed_lines.append("  Total:")
            detailed_lines.extend(f"    {typ}: {n}"
                                  for typ, n in sorted(message_counts[sender]["Total"].items()))
            detailed_lines.append("  Replies to:")
            detailed_lines.extend(f"    {receiver}: {n}" for receiver, n in
                                  sorted(reply_counts[sender].items()))
            detailed_lines.append("")
        format_text_pages(pdf, "\n".join(detailed_lines), "Detailed LC Message Counts")


    return pdf_buffer.getvalue(), {
        "n_blocks": n_blocks,
        "unique_messages": len(records),
        "components": component_count,
        "explicit_replies": explicit_replies,
        "inferred_replies": inferred_replies,
        "skipped": dict(skipped),
        "first_date": earliest_timestamp,
        "last_date": latest_timestamp,
        "word_totals": dict(sender_word_totals),
        "average_words": sender_avg_words,
    }
