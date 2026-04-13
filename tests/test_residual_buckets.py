import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from simple_multistep_model import DataFrameMultistepModel, ResidualBootstrapModel
from simple_multistep_model.multistep import _period_token_from_string
from simple_multistep_model.one_step_model import ResidualDistribution, extract_period_token


class ResidualBucketTests(unittest.TestCase):
    def test_extract_period_token(self):
        self.assertEqual(extract_period_token("2026-03"), "03")
        self.assertEqual(extract_period_token("2026-W07"), "W07")

    def test_period_token_from_string(self):
        self.assertEqual(_period_token_from_string("2026-11"), "11")
        self.assertEqual(_period_token_from_string("2026-W52"), "W52")

    def test_distribution_fallback_order(self):
        # Row 0: has location+period pool with size >=5 -> use that pool
        # Row 1: location+period pool missing -> fallback to location pool
        # Row 2: location pool missing -> fallback to global pool
        dist = ResidualDistribution(
            predictions=np.array([0.0, 0.0, 0.0]),
            residuals=np.array([100.0, 101.0]),
            residuals_by_location={
                "A": np.array([10.0, 11.0]),
            },
            residuals_by_location_period={
                ("A", "01"): np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            },
            min_bucket_size=5,
        )

        samples = dist.sample(
            50,
            context_by_row=[("A", "01"), ("A", "02"), ("B", "01")],
        )

        self.assertTrue(np.isin(samples[:, 0], np.array([1.0, 2.0, 3.0, 4.0, 5.0])).all())
        self.assertTrue(np.isin(samples[:, 1], np.array([10.0, 11.0])).all())
        self.assertTrue(np.isin(samples[:, 2], np.array([100.0, 101.0])).all())

    def test_fit_populates_location_period_buckets(self):
        rows = []
        for loc in ["A", "B"]:
            for month in range(1, 13):
                rows.append(
                    {
                        "time_period": f"2025-{month:02d}",
                        "location": loc,
                        "disease_cases": float(20 + month + (0 if loc == "A" else 2)),
                        "smc_number": float(month % 3),
                        "rainfall": float(month),
                        "mean_temperature": 25.0,
                        "rel_humidity": 0.5,
                        "population": 1000.0,
                        "area": 10.0,
                        "median_elevation": 100.0,
                    }
                )

        df = pd.DataFrame(rows)
        X = df[
            [
                "time_period",
                "location",
                "smc_number",
                "rainfall",
                "mean_temperature",
                "rel_humidity",
                "population",
                "area",
                "median_elevation",
            ]
        ]
        y = df[["time_period", "location", "disease_cases"]]

        one_step = ResidualBootstrapModel(
            RandomForestRegressor(n_estimators=10, random_state=0),
            min_bucket_size=5,
        )
        model = DataFrameMultistepModel(one_step, n_target_lags=3, target_variable="disease_cases")
        model.fit(X, y)

        self.assertIn("A", one_step._residuals_by_location)
        self.assertIn(("A", "04"), one_step._residuals_by_location_period)
        self.assertIn(("B", "12"), one_step._residuals_by_location_period)


if __name__ == "__main__":
    unittest.main()
