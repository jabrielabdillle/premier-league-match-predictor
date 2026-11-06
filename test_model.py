import unittest
import pandas as pd
from model import make_features


class FeatureTests(unittest.TestCase):
    def test_same_day_results_are_excluded(self):
        matches = pd.DataFrame([
            ["2024-08-01", "A", "B", 2, 0, "H"],
            ["2024-08-01", "C", "A", 3, 0, "H"],
            ["2024-08-02", "A", "D", 0, 0, "D"],
        ], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        first = make_features(matches)
        self.assertEqual(first.loc[1, "away_history"], 0)
        self.assertEqual(first.loc[2, "home_points"], 1.5)
        matches.loc[2, ["FTHG", "FTAG", "FTR"]] = [1, 0, "H"]
        pd.testing.assert_frame_equal(first, make_features(matches))
