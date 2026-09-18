"""Offline tests using invented values, never actual market observations."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from ml.valuation.ingestion import SOURCE_COLUMNS, ingest, load_csv, summarize_numbers
from ml.valuation.parsing import (
    normalize_text, parse_engine_capacity, parse_engine_with_unit,
    parse_mileage, parse_model_year, parse_price,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "data/fixtures/ingestion_synthetic.json").read_text(encoding="utf-8"))
RECEIPT = json.loads((ROOT / "data/manifests/mendeley_fmb4xmp4k5_v2.json").read_text(encoding="utf-8"))


class ParsingTests(unittest.TestCase):
    def test_price_units_and_exact_integer_conversion(self):
        cases = {
            "BDT 2,250,000": 2250000, "৳22,50,000": 2250000,
            "Tk. 2250000/-": 2250000, "2250000 টাকা": 2250000,
            "22.5 lakh": 2250000, "২২.৫ লাখ": 2250000,
            "BDT 1.25 crore": 12500000, "1 crore BDT": 10000000,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = parse_price(text)
                self.assertEqual(parsed.value, expected)
                self.assertIsInstance(parsed.value, int)
                self.assertIsNone(parsed.issue)

    def test_price_requires_evidence_for_bare_units(self):
        for text in ("2250000", "22.5"):
            self.assertEqual(parse_price(text).issue, "ambiguous_price_unit")
        self.assertEqual(parse_price("2250000", bare_unit="BDT").value, 2250000)
        self.assertEqual(parse_price("22.5", bare_unit="lakh_BDT").value, 2250000)
        with self.assertRaises(ValueError):
            parse_price("10", bare_unit="INR")

    def test_finance_and_foreign_currency_are_not_asking_prices(self):
        for text in ("BDT 50000 deposit", "down payment 5 lakh", "Tk 5000 monthly", "20 lakh EMI", "৫ লাখ অগ্রিম"):
            with self.subTest(text=text):
                self.assertEqual(parse_price(text).issue, "non_full_price")
        for text in ("USD 20 lakh", "₹22,50,000", "$50000", "2250000 INR"):
            self.assertEqual(parse_price(text).issue, "unsupported_currency")

    def test_invalid_prices_do_not_get_rounded_or_repaired(self):
        cases = {
            "BDT -1": "negative_value", "-22.5 lakh": "negative_value",
            "BDT 0": "zero_not_allowed", "BDT 1.5": "fractional_normalized_value",
            "BDT 2,250,00": "invalid_number_format", "BDT 1e6": "invalid_number_format",
            "22-25 lakh": "invalid_number_format", "negotiable": "invalid_number_format",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                parsed = parse_price(text)
                self.assertIsNone(parsed.value)
                self.assertEqual(parsed.issue, reason)
        # Decimal's default precision must not silently round a long source value.
        self.assertEqual(parse_price("BDT 123456789012345678901234567890.1").issue, "fractional_normalized_value")

    def test_mileage_units_zero_and_invalid_values(self):
        for text in ("65000", "65,000 km", "৬৫,০০০ kilometres"):
            self.assertEqual(parse_mileage(text).value, 65000)
        self.assertEqual(parse_mileage("0 km").value, 0)
        for text, reason in (("-1 km", "negative_value"), ("65k", "invalid_number_format"), ("10 miles", "invalid_number_format"), ("1.5 km", "fractional_normalized_value")):
            self.assertEqual(parse_mileage(text).issue, reason)

    def test_engine_units_and_ambiguity(self):
        for text in ("1500 cc", "1,500 cm³", "1.5 litres", "১৫০০ সিসি"):
            self.assertEqual(parse_engine_capacity(text).value, 1500)
        for text in ("1500", "1.5"):
            self.assertEqual(parse_engine_capacity(text).issue, "ambiguous_engine_unit")
        self.assertEqual(parse_engine_with_unit("1500", bare_unit="cc").value, 1500)
        self.assertEqual(parse_engine_with_unit("1.5", bare_unit="cc").issue, "fractional_normalized_value")
        for text in ("-1500", "-1.5 L"):
            self.assertEqual(parse_engine_capacity(text).issue, "negative_value")
        self.assertEqual(parse_engine_capacity("0 cc").issue, "zero_not_allowed")
        with self.assertRaises(ValueError):
            parse_engine_with_unit("1500 cc", bare_unit="litres")

    def test_nulls_and_booleans(self):
        for parser in (parse_price, parse_mileage, parse_engine_capacity, parse_model_year):
            for value in (None, "", "  ", "N/A", "null", "unknown", "-"):
                with self.subTest(parser=parser.__name__, value=value):
                    self.assertIsNone(parser(value).value)
                    self.assertEqual(parser(value).issue, "missing")
            self.assertEqual(parser(True).issue, "invalid_type")

    def test_text_normalization_and_year(self):
        self.assertEqual(normalize_text("  Example\u00a0 Motors\n "), "Example Motors")
        self.assertIsNone(normalize_text(" N/A "))
        self.assertEqual(parse_model_year("২০১৮").value, 2018)
        self.assertEqual(parse_model_year("2018.5").issue, "fractional_normalized_value")
        self.assertEqual(parse_model_year("10000").issue, "invalid_year")


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bd-synthetic-ingestion-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.raw = self.directory / "synthetic.csv"
        self.rows = copy.deepcopy(FIXTURE["rows"])
        self.receipt = copy.deepcopy(RECEIPT)
        self.receipt.update({
            "dataset_id": "synthetic-test-only", "dataset_name": "Synthetic test cases",
            "source_url": "https://example.invalid/synthetic", "doi": None,
            "raw_status": "SYNTHETIC_TEST_ONLY", "retrieved_on": "2026-09-18",
            "original_filename": self.raw.name,
            "publisher_reported_row_count": len(self.rows),
        })
        self.write_raw()

    def write_raw(self, headers=SOURCE_COLUMNS):
        with self.raw.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(self.rows)
        self.refresh_hash()

    def refresh_hash(self):
        self.receipt["file_sha256"] = hashlib.sha256(self.raw.read_bytes()).hexdigest()

    def run_ingestion(self, name="output"):
        output = self.directory / name
        report = ingest(self.raw, self.receipt, output)
        normalized = [json.loads(line) for line in (output / "normalized.jsonl").read_text(encoding="utf-8").splitlines()]
        return report, normalized

    def test_all_rows_preserved_with_correct_synthetic_counts(self):
        report, rows = self.run_ingestion()
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(report["raw_row_count"], 9)
        self.assertEqual(report["usable_row_count"], 2)
        self.assertEqual(report["duplicate_count"], 1)
        self.assertEqual(report["normalized_duplicate_count"], 1)
        self.assertEqual(report["invalid_price_count"], 3)
        self.assertEqual(report["suspicious_price_count"], 1)
        self.assertEqual(report["missing_price_count"], 1)
        self.assertEqual(report["missing_values_per_column"]["engine_capacity"], 1)
        self.assertEqual(report["actual_column_names"], list(SOURCE_COLUMNS))
        self.assertEqual(rows[1]["exact_duplicate_of"], 1)
        self.assertEqual(rows[2]["normalized_duplicate_of"], 1)
        self.assertFalse(rows[1]["usable_for_historical_baseline"])
        self.assertEqual(report["numeric_summary_usable_rows"]["price"]["max"], 2250000)

    def test_provenance_raw_values_and_schema_compatibility(self):
        _, rows = self.run_ingestion()
        schema = json.loads((ROOT / "data/schemas/market_record.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        for row in rows:
            source = self.rows[row["source_row_number"] - 1]
            for record in row["records"]:
                validator.validate(record)
                provenance = record["provenance"]
                self.assertEqual(provenance["source_id"], "synthetic-test-only")
                self.assertEqual(provenance["raw_values"], {key: value if value is not None else "" for key, value in source.items()})
                self.assertEqual(provenance["snapshot_ref"], "sha256:" + self.receipt["file_sha256"])
                self.assertIsNone(provenance["collected_on"])
                self.assertEqual(provenance["collection_date_status"], "unknown")
                self.assertIn(":row:" + str(row["source_row_number"]), provenance["source_record_ref"])
            vehicle, listing, observation = row["records"]
            self.assertEqual(listing["vehicle_id"], vehicle["id"])
            self.assertEqual(observation["listing_id"], listing["id"])
            self.assertIsNone(listing["listed_on"])
            self.assertIsNone(listing["market_category"])
            self.assertIsNone(observation["observed_on"])
            self.assertIsNone(vehicle["hybrid_status"])
        self.assertEqual(rows[2]["records"][0]["make"], "example motors")

    def test_conflicting_title_year_is_retained_but_not_trained(self):
        self.rows = [copy.deepcopy(self.rows[0])]
        self.rows[0].update(car_name="Example Motors Sample 2011", model_year="2016")
        self.write_raw()
        report, rows = self.run_ingestion()
        self.assertEqual(report["usable_row_count"], 0)
        self.assertIn("title_year_conflict", [issue["reason"] for issue in rows[0]["issues"]])
        self.assertEqual(rows[0]["records"][0]["model_year"], 2016)
        self.assertEqual(rows[0]["records"][0]["provenance"]["raw_values"]["car_name"], "Example Motors Sample 2011")

    def test_missing_negative_finance_and_outlier_rows_remain_flagged(self):
        _, rows = self.run_ingestion()
        missing = rows[4]["records"]
        self.assertIsNone(missing[0]["engine_cc"])
        self.assertIsNone(missing[2]["mileage_km"])
        self.assertIsNone(missing[2]["asking_price_bdt"])
        negative = rows[5]
        self.assertEqual({i["field"] for i in negative["issues"] if i["reason"] == "negative_value"}, {"price", "kilometers_run", "engine_capacity"})
        self.assertIsNone(negative["records"][2]["asking_price_bdt"])
        self.assertEqual(rows[6]["records"][2]["amount_kind"], "unknown")
        self.assertIn("non_full_price", [i["reason"] for i in rows[6]["issues"]])
        self.assertEqual(rows[7]["records"][2]["asking_price_bdt"], 150000000)
        self.assertEqual(sum(i["reason"] == "outside_review_range" for i in rows[7]["issues"]), 4)

    def test_reproducible_outputs_and_unchanged_raw_bytes(self):
        before = self.raw.read_bytes()
        self.run_ingestion("first")
        self.run_ingestion("second")
        for name in ("normalized.jsonl", "quality_report.json", "run_manifest.json"):
            self.assertEqual((self.directory / "first" / name).read_bytes(), (self.directory / "second" / name).read_bytes())
        self.assertEqual(self.raw.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.run_ingestion("first")
        self.assertEqual(self.raw.read_bytes(), before)

    def test_receipt_guards_prevent_unverified_ingestion(self):
        cases = (
            ("file_sha256", None), ("file_sha256", "0" * 64),
            ("original_filename", "other.csv"), ("retrieved_on", None),
            ("retrieved_on", "not-a-date"), ("market", "IN"),
            ("eligibility_status", "PERMISSION_REQUIRED"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                receipt = copy.deepcopy(self.receipt)
                receipt[field] = value
                with self.assertRaises(ValueError):
                    ingest(self.raw, receipt, self.directory / "rejected")
                self.assertFalse((self.directory / "rejected").exists())

    def test_bare_units_cannot_be_configured_without_evidence(self):
        for field, value in (("bare_price", "BDT"), ("bare_engine", "cc")):
            receipt = copy.deepcopy(self.receipt)
            receipt["normalization"][field + "_unit"] = value
            with self.assertRaisesRegex(ValueError, "evidence"):
                ingest(self.raw, receipt, self.directory / "no-evidence")
        self.receipt["normalization"].update({
            "bare_price_unit": "BDT", "bare_price_unit_evidence": "Synthetic fixture author's explicit convention; not source evidence",
            "bare_engine_unit": "cc", "bare_engine_unit_evidence": "Synthetic fixture author's explicit convention; not source evidence",
        })
        _, rows = self.run_ingestion()
        self.assertEqual(rows[8]["records"][2]["asking_price_bdt"], 2250000)
        self.assertEqual(rows[8]["records"][0]["engine_cc"], 1500)

    def test_unknown_and_extra_columns_are_preserved(self):
        for row in self.rows:
            row["unmapped_note"] = "Invented test note"
        self.write_raw((*SOURCE_COLUMNS, "unmapped_note"))
        report, rows = self.run_ingestion()
        self.assertEqual(report["extra_columns"], ["unmapped_note"])
        self.assertEqual(rows[0]["records"][0]["provenance"]["raw_values"]["unmapped_note"], "Invented test note")

    def test_wrong_width_rows_are_retained(self):
        with self.raw.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(["partial", "Example Motors"])
        self.refresh_hash()
        report, rows = self.run_ingestion()
        self.assertEqual(report["raw_row_count"], 10)
        self.assertIn("column_count_mismatch", [i["reason"] for i in rows[-1]["issues"]])

    def test_bad_headers_fail_before_output(self):
        for headers in (("price", " PRICE "), ("brand", "price")):
            with self.raw.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(headers)
            self.refresh_hash()
            with self.assertRaises(ValueError):
                self.run_ingestion()
            self.assertFalse((self.directory / "output").exists())

    def test_distinct_failed_parses_are_not_duplicate_matches(self):
        self.rows = [copy.deepcopy(self.rows[0]), copy.deepcopy(self.rows[0])]
        self.rows[0]["price"] = "unknown format one"
        self.rows[1]["price"] = "unknown format two"
        self.write_raw()
        report, _ = self.run_ingestion()
        self.assertEqual(report["normalized_duplicate_count"], 0)

    def test_report_empty_data_and_numeric_summaries(self):
        self.rows = []
        self.write_raw()
        report, rows = self.run_ingestion()
        self.assertEqual(rows, [])
        self.assertEqual(report["raw_row_count"], 0)
        self.assertIsNone(report["numeric_summary_all_parseable_rows"]["price"]["min"])
        summary = summarize_numbers([0, 100, None])
        self.assertEqual(summary["median"], 50)
        self.assertEqual(summary["p05"], 5)


if __name__ == "__main__":
    unittest.main()
