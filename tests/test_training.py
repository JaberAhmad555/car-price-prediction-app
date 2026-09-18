"""Exercise ML mechanics using synthetic data, never report these as market metrics."""
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ml.valuation.train import FEATURES, fit_candidates


class TrainingTests(unittest.TestCase):
    def test_candidates_fit_and_duplicate_profiles_stay_together(self):
        rows = [{"brand": "test", "model": "test-a" if i % 2 else "test-b", "year": 2000 + i % 20,
                 "mileage_km": i * 1000, "engine_cc": 1500, "transmission": "automatic",
                 "fuel_type": "petrol", "body_type": "sedan", "price_bdt": 1000000 + i * 7000} for i in range(70)]
        rows.append({**rows[0], "price_bdt": 1100000})
        frame = pd.DataFrame(rows)
        with threadpool_limits(limits=2):
            pipeline, report = fit_candidates(frame)
        self.assertEqual(len(report["validation_metrics"]), 3)
        self.assertTrue(all(np.isfinite(value) for value in report["test_metrics"].values()))
        splits = report["split_indices"]
        self.assertEqual(sum(map(len, splits.values())), len(frame))
        for indexes in splits.values():
            self.assertEqual(0 in indexes, 70 in indexes)
        self.assertEqual(pipeline.predict(frame.iloc[:1][FEATURES]).shape, (1,))
        self.assertEqual(report["selected_model"], min(report["validation_metrics"], key=lambda name: report["validation_metrics"][name]["mae_bdt"]))

    def test_tiny_dataset_cannot_claim_evaluation(self):
        frame = pd.DataFrame([{**{key: "test" for key in FEATURES}, "price_bdt": 1000}])
        with self.assertRaisesRegex(ValueError, "30 distinct"):
            fit_candidates(frame)
