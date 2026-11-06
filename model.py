"""Evaluate pre-match Premier League probabilities on a held-out season."""
import argparse
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = ("2122", "2223", "2324", "2425")
BASE = "https://www.football-data.co.uk/mmz4281"
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
LABELS = {"H": "home_win", "D": "draw", "A": "away_win"}


def load_matches(data_dir):
    frames = []
    for season in SEASONS:
        source = data_dir / f"{season}.csv" if data_dir else f"{BASE}/{season}/E0.csv"
        frame = pd.read_csv(source, usecols=COLUMNS)
        frame["Date"] = pd.to_datetime(frame["Date"], format="%d/%m/%Y")
        expected = frame.FTHG.gt(frame.FTAG).map({True: "H", False: "A"})
        expected = expected.mask(frame.FTHG.eq(frame.FTAG), "D")
        if (len(frame) != 380 or frame[COLUMNS].isna().any().any()
                or frame.duplicated(["HomeTeam", "AwayTeam"]).any()
                or not expected.eq(frame.FTR).all()):
            raise ValueError(f"Unexpected or incomplete match data for {season}")
        frames.append(frame.assign(season=season))
    return pd.concat(frames, ignore_index=True).sort_values("Date", kind="stable")


def make_features(matches):
    """Use each club's last five appearances on strictly earlier dates."""
    history = defaultdict(lambda: deque(maxlen=5))
    rows = []
    for _, day in matches.sort_values("Date", kind="stable").groupby("Date"):
        for match in day.itertuples():
            row = {}
            for side, team in (("home", match.HomeTeam), ("away", match.AwayTeam)):
                form = history[team]
                row[f"{side}_history"] = len(form)
                for i, name in enumerate(("points", "goals_for", "goals_against")):

                    row[f"{side}_{name}"] = sum(game[i] for game in form) / len(form) if form else 1.0
            rows.append((match.Index, row))
        for match in day.itertuples():
            home_points, away_points = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[match.FTR]
            history[match.HomeTeam].append((home_points, match.FTHG, match.FTAG))
            history[match.AwayTeam].append((away_points, match.FTAG, match.FTHG))
    return pd.DataFrame.from_dict(dict(rows), orient="index").reindex(matches.index)


def run(data_dir, output):
    matches = load_matches(data_dir)
    features = make_features(matches)
    train = matches.season.isin(SEASONS[1:-1])
    test = matches.season.eq(SEASONS[-1])
    predictions = matches.loc[test, ["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    scores = []
    models = {
        "class_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }
    for name, model in models.items():
        model.fit(features.loc[train], matches.loc[train, "FTR"])
        probabilities = model.predict_proba(features.loc[test])
        predicted = model.predict(features.loc[test])
        actual = matches.loc[test, "FTR"]
        scores.append({"model": name, "accuracy": accuracy_score(actual, predicted),
                       "log_loss": log_loss(actual, probabilities, labels=model.classes_)})
        if name == "logistic_regression":
            predictions["predicted"] = predicted
            for i, label in enumerate(model.classes_):
                predictions[f"p_{LABELS[label]}"] = probabilities[:, i]
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(scores)
    summary.to_csv(output / "metrics.csv", index=False, float_format="%.6f")
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.8f")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data_dir, args.output)
