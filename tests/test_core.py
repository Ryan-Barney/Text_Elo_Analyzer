"""Run: python -m unittest discover -s tests -v"""
import unittest
from collections import Counter
from pathlib import Path
from analysis_core import (
    analyze, generate_pdf, load_records_from_text, parse_common_words_csv,
    relabel_records, word_token,
)

DIV = "-" * 52

class ParserTests(unittest.TestCase):
    def test_mixed_formats_and_uuid_components(self):
        sample = f"""{DIV}\nTest chat\n2026-05-01 12:01:02 from Alice Example (+100000) - Read\n\nHello world\n{DIV}\nTest chat\n12345678-1234-1234-1234-123456789012\n2026-05-01 12:02:00 to Test chat - Sent\n\nimage.jpg (Image)\n{DIV}\nTest chat\n12345678-1234-1234-1234-123456789012\n2026-05-01 12:02:00 to Test chat - Sent\n\nAn image caption here\n{DIV}\n"""
        records, skips, blocks = load_records_from_text(sample)
        self.assertEqual((len(records), blocks), (2, 3))
        self.assertEqual(len(records[-1]["components"]), 2)
        self.assertEqual(records[-1]["sender"], "You")
        mapping = {"You": "Me", "Alice Example": "Alice"}
        mcounts, hourly, *_ = analyze(relabel_records(records, mapping), mapping)
        self.assertEqual(mcounts["Me"]["Total"]["Image"], 1)
        self.assertEqual(mcounts["Me"]["Total"]["Text"], 1)
        self.assertEqual(hourly["Me"][12], 1)

    def test_word_token_and_csv(self):
        self.assertEqual(word_token("I’m"), "I'M")
        self.assertEqual(word_token("🟩🟩🟩🟩🟩"), "🟩🟩🟩🟩🟩")
        self.assertEqual(parse_common_words_csv(b"word\nI'm\nand\n"), {"I'M", "AND"})

    def test_pdf_bytes_generated(self):
        sample = (Path(__file__).resolve().parents[1] / 'sample_export.txt').read_text()
        records, skip, blocks = load_records_from_text(sample)
        names = {r["sender"] for r in records}
        names_map = {name: ("Me" if name == "You" else name) for name in names}
        style = {name: {"color": "#336699", "marker": "o"}
                 for name in names_map.values()}
        pdf, meta = generate_pdf(records, skip, blocks, names_map, style)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 10000)
        self.assertEqual(meta["unique_messages"], len(records))

if __name__ == "__main__":
    unittest.main()
