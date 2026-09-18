"""Contract tests using synthetic records only; no data acquisition or ML."""

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (ROOT / "data/schemas/market_record.schema.json").read_text(encoding="utf-8")
)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def provenance():
    return {
        "source_id": "synthetic-source",
        "source_record_ref": "synthetic-row-1",
        "collected_on": None,
        "collection_date_status": "unknown",
        "raw_values": {"fixture_notice": "Synthetic test only; not a market record"},
        "normalization_version": "test-only-0.1.0",
    }


def example_records():
    """Return fresh synthetic entities; monetary values are test inputs, not estimates."""
    fields = {
        "data_source": {
            "provider": "Synthetic test provider",
            "source_reference": "test-only:not-a-real-data-source",
            "acquisition_method": "manual_curation",
            "license_terms_ref": None,
            "reviewed_on": None,
            "review_evidence_ref": None,
            "permissions": {
                name: "unknown"
                for name in (
                    "storage", "automated_collection", "ml_training",
                    "derived_predictions", "public_display", "redistribution",
                )
            },
            "attribution_requirements": None,
            "restrictions": None,
            "eligibility": "pending_review",
        },
        "vehicle": {
            "make": "Synthetic Make", "model": "Synthetic Model",
            "variant": None, "hybrid_status": None, "manufacture_year": None,
        },
        "listing": {
            "vehicle_id": "synthetic-vehicle", "market_category": "Local Used",
            "listed_on": None, "location": None,
        },
        "listing_observation": {
            "listing_id": "synthetic-listing", "observed_on": None,
            "currency": "BDT", "price_type": "asking",
            "amount_kind": "full_asking_price", "asking_price_bdt": 1234567,
            "mileage_km": None,
        },
        "condition_observation": {
            "vehicle_id": "synthetic-vehicle", "observed_on": None,
            "evidence_status": "unverified", "reported_condition": None,
            "accident_history": None, "previous_owners": None,
        },
        "registration_observation": {
            "vehicle_id": "synthetic-vehicle", "observed_on": None,
            "evidence_status": "unverified", "registration_status": None,
            "registration_year": None, "tax_token_valid_until": None,
            "fitness_valid_until": None,
        },
    }
    records = {}
    for entity, values in fields.items():
        record = {
            "schema_version": "0.1.0", "entity_type": entity,
            "id": "synthetic-" + entity, "market": "BD", **values,
        }
        if entity != "data_source":
            record["provenance"] = provenance()
        records[entity] = record
    return records


def approved_source():
    """A synthetic declaration for schema checks; it approves no real source."""
    source = example_records()["data_source"]
    source.update({
        "eligibility": "approved_for_valuation",
        "reviewed_on": "2026-01-01",
        "license_terms_ref": "Synthetic test agreement",
        "review_evidence_ref": "Synthetic test review",
        "attribution_requirements": "Synthetic test attribution",
        "restrictions": "Synthetic test restriction statement",
    })
    for use in ("storage", "ml_training", "derived_predictions"):
        source["permissions"][use] = "permitted"
    return source


class MarketRecordSchemaTests(unittest.TestCase):
    def assert_invalid(self, record):
        with self.assertRaises(ValidationError):
            VALIDATOR.validate(record)

    def test_schema_is_valid_draft_2020_12(self):
        Draft202012Validator.check_schema(SCHEMA)

    def test_all_six_entities_validate(self):
        for entity, record in example_records().items():
            with self.subTest(entity=entity):
                VALIDATOR.validate(record)

    def test_optional_unknowns_remain_null_and_records_are_not_mutated(self):
        for entity, record in example_records().items():
            before = copy.deepcopy(record)
            with self.subTest(entity=entity):
                VALIDATOR.validate(record)
                self.assertEqual(record, before)
        observation = example_records()["listing_observation"]
        observation["asking_price_bdt"] = None
        VALIDATOR.validate(observation)
        self.assertIsNone(observation["asking_price_bdt"])

    def test_invalid_price_values_are_rejected_without_coercion(self):
        for value in (-1, -1000000, 0, 123.45, "1234567", True, False):
            with self.subTest(value=value):
                record = example_records()["listing_observation"]
                record["asking_price_bdt"] = value
                self.assert_invalid(record)

    def test_full_bdt_amount_is_preserved(self):
        record = example_records()["listing_observation"]
        VALIDATOR.validate(record)
        self.assertEqual(record["asking_price_bdt"], 1234567)

    def test_only_bangladesh_market_and_bdt_currency_are_accepted(self):
        for entity, record in example_records().items():
            with self.subTest(entity=entity):
                record["market"] = "OTHER"
                self.assert_invalid(record)
        for currency in ("USD", "INR", None):
            with self.subTest(currency=currency):
                record = example_records()["listing_observation"]
                record["currency"] = currency
                self.assert_invalid(record)

    def test_transaction_price_cannot_replace_or_join_asking_price(self):
        record = example_records()["listing_observation"]
        record["price_type"] = "confirmed_transaction"
        self.assert_invalid(record)
        record = example_records()["listing_observation"]
        record["confirmed_transaction_price_bdt"] = 1234567
        self.assert_invalid(record)

    def test_non_full_amounts_never_become_asking_price_labels(self):
        for kind in ("deposit", "monthly_payment", "unknown"):
            with self.subTest(kind=kind):
                record = example_records()["listing_observation"]
                record["amount_kind"] = kind
                self.assert_invalid(record)
                record["asking_price_bdt"] = None
                VALIDATOR.validate(record)

    def test_provenance_and_raw_values_are_required(self):
        for entity, record in example_records().items():
            if entity == "data_source":
                continue
            with self.subTest(entity=entity):
                del record["provenance"]
                self.assert_invalid(record)
        for field in provenance():
            with self.subTest(missing_field=field):
                record = example_records()["vehicle"]
                del record["provenance"][field]
                self.assert_invalid(record)
        record = example_records()["vehicle"]
        record["provenance"]["raw_values"] = {}
        self.assert_invalid(record)

    def test_original_values_survive_validation(self):
        record = example_records()["listing"]
        raw = {"category": "Example original source label", "year": "not supplied"}
        record["provenance"]["raw_values"] = copy.deepcopy(raw)
        VALIDATOR.validate(record)
        self.assertEqual(record["provenance"]["raw_values"], raw)

    def test_collection_date_status_and_calendar_date_are_enforced(self):
        record = example_records()["vehicle"]
        record["provenance"]["collection_date_status"] = "known"
        self.assert_invalid(record)
        for value in ("2026-02-30", "2026-2-01", "not-a-date"):
            with self.subTest(value=value):
                record["provenance"]["collected_on"] = value
                self.assert_invalid(record)
        record["provenance"]["collected_on"] = "2026-02-28"
        VALIDATOR.validate(record)
        record["provenance"]["collection_date_status"] = "unknown"
        self.assert_invalid(record)

    def test_market_category_is_not_physical_condition(self):
        listing = example_records()["listing"]
        for category in ("Local Used", "Foreign Used", "Reconditioned", "Brand New", None):
            with self.subTest(category=category):
                listing["market_category"] = category
                VALIDATOR.validate(listing)
        listing["market_category"] = "Severe damage"
        self.assert_invalid(listing)
        condition = example_records()["condition_observation"]
        condition["market_category"] = "Reconditioned"
        self.assert_invalid(condition)

    def test_known_zero_counts_are_allowed_but_negative_counts_are_not(self):
        for entity, field in (
            ("listing_observation", "mileage_km"),
            ("condition_observation", "previous_owners"),
        ):
            with self.subTest(field=field):
                record = example_records()[entity]
                record[field] = 0
                VALIDATOR.validate(record)
                record[field] = -1
                self.assert_invalid(record)

    def test_unregistered_vehicles_do_not_have_a_registration_year(self):
        record = example_records()["registration_observation"]
        record["registration_status"] = "unregistered"
        VALIDATOR.validate(record)
        for value in (0, 2020):
            with self.subTest(value=value):
                record["registration_year"] = value
                self.assert_invalid(record)

    def test_present_price_and_unrecognized_fields_are_rejected(self):
        for field in ("Present Price", "Present_Price", "present_price", "manufacturing_yaer"):
            with self.subTest(field=field):
                record = example_records()["vehicle"]
                record[field] = 123
                self.assert_invalid(record)

    def test_source_approval_requires_permissions_and_review_evidence(self):
        VALIDATOR.validate(approved_source())
        for permission in ("storage", "ml_training", "derived_predictions"):
            for value in ("unknown", "prohibited"):
                with self.subTest(permission=permission, value=value):
                    source = approved_source()
                    source["permissions"][permission] = value
                    self.assert_invalid(source)
        for field in (
            "reviewed_on", "license_terms_ref", "review_evidence_ref",
            "attribution_requirements", "restrictions",
        ):
            with self.subTest(missing_evidence=field):
                source = approved_source()
                source[field] = None
                self.assert_invalid(source)

    def test_automated_acquisition_requires_separate_permission(self):
        for method in ("api", "automated_collection"):
            with self.subTest(method=method):
                source = approved_source()
                source["acquisition_method"] = method
                self.assert_invalid(source)
                source["permissions"]["automated_collection"] = "permitted"
                VALIDATOR.validate(source)


if __name__ == "__main__":
    unittest.main()
