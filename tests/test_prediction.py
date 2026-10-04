import unittest

from services.prediction import (
    FEATURE_COLUMNS,
    classifier_global_drivers,
    predict_site,
    prepare_input,
)


SAMPLE = dict(
    zip(
        FEATURE_COLUMNS,
        (
            0.99,
            16.08,
            16.60,
            53.40,
            0.55,
            0.62,
            3.39,
            0.48,
            1.40,
            0.69,
            0.20,
            1373,
            "lowland",
            "Pasture",
        ),
        strict=True,
    )
)


class PredictionServiceTests(unittest.TestCase):
    def test_prediction_includes_uncertainty_and_vote_share(self):
        result = predict_site(SAMPLE)

        self.assertGreaterEqual(result["predicted_median"], 0)
        self.assertLessEqual(
            result["regression_interval"]["lower"],
            result["predicted_median"],
        )
        self.assertGreaterEqual(
            result["regression_interval"]["upper"],
            result["predicted_median"],
        )
        self.assertAlmostEqual(
            result["predicted_band_vote_share"],
            result["band_scores"][result["predicted_band"]],
        )
        self.assertAlmostEqual(sum(result["band_scores"].values()), 1.0)

    def test_lowland_pasture_receives_reliability_warning(self):
        result = predict_site(SAMPLE)
        self.assertTrue(result["reliability_notes"])

    def test_land_cover_total_over_100_is_rejected(self):
        invalid = SAMPLE | {"pct_urban_area": 50.0}
        with self.assertRaisesRegex(ValueError, "total"):
            prepare_input(invalid)

    def test_global_driver_summary_has_three_features(self):
        drivers = classifier_global_drivers()
        self.assertEqual(len(drivers), 3)
        self.assertTrue(all(driver["importance"] > 0 for driver in drivers))


if __name__ == "__main__":
    unittest.main()
