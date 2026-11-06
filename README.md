# Premier League Match Predictor

Can a team's recent form improve pre-match predictions over the usual home-win baseline? This project estimates three probabilities for each fixture: home win, draw, and away win.

## Held-out results

| Model | Accuracy | Log loss |
| --- | ---: | ---: |
| Historical class frequencies | 40.8% | 1.086 |
| Logistic regression | 47.9% | 1.045 |

Accuracy improved by 7.1 percentage points across the 380 matches in 2024/25. Lower log loss indicates better probability predictions. The model assigned a probability to draws, but a draw was never its highest-probability outcome. The test season contained 93 draws, making draw prediction a clear area for improvement.

## Method

- Source: [Football-Data.co.uk, England match results](https://www.football-data.co.uk/englandm/index.php), four complete Premier League seasons from 2021/22 through 2024/25.
- Features: average points, goals scored, and goals conceded over each club's previous five Premier League appearances, plus the number of appearances available. Histories carry across seasons. Clubs with no history use one point and one goal for and against as defaults.
- Timeline: 2021/22 warms up team histories. Training uses 2022/23 and 2023/24. Evaluation uses 2024/25. Earlier results in the test season update form as they become available. Model coefficients remain fixed.
- Leakage control: every match on a given date is scored using results from strictly earlier dates. Current match scores and betting odds are excluded from the predictors. The scaler is fit only on training data.

## Run on macOS

From the extracted folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python model.py
```

The script downloads the four CSVs. For an offline rerun, place them in one directory as `2122.csv`, `2223.csv`, `2324.csv`, and `2425.csv`, then pass `--data-dir /path/to/files`. Open `outputs/predictions.csv` for match probabilities and `outputs/metrics.csv` for the comparison. The verified run's aggregate scores are in `example/metrics.csv`.
