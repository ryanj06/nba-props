# 🏀 NBA Props Predictor

AI-powered NBA player prop predictions. Live at [nba-props.streamlit.app](https://nba-props.streamlit.app).

---

## What it does

You pick a game, pick a team, pick a player — the app pulls their recent stats and tonight's matchup context, runs it through an XGBoost model, and tells you the probability of a prop line hitting with a betting recommendation.

Supports points, rebounds, assists, 3PT made, steals, blocks, turnovers, PRA, P+R, and P+A.

---

## Why I built it

Most people bet props based on season averages. But a player averaging 25 points hits differently against a top-5 defense on a back-to-back road game than they do against a weak defense at home rested. I wanted to build something that actually accounts for that context.

---

## Tech stack

- **Data** — nba_api (pulls live game logs, team defensive stats, rosters)
- **Model** — XGBoost regression, one model per stat category
- **Validation** — TimeSeriesSplit (5 folds, always trains on past, validates on future)
- **App** — Streamlit with custom dark CSS
- **Deployment** — Streamlit Community Cloud
- **Automation** — GitHub Actions daily retraining

---

## How the model works

27,000+ player-game rows across 3 seasons (2023-24, 2024-25, 2025-26). 20 features per game including rolling averages, season averages, home/away, back-to-back, opponent defensive rating, and pace.

The biggest thing I had to get right was preventing data leakage. Rolling averages need to be calculated from games *before* the one you're predicting — if you include the current game's stats in "last 5 game average" the model looks great in training and falls apart on real predictions. Fixed by using `shift(1)` before every rolling window.

Used TimeSeriesSplit instead of random cross-validation because random splits let the model train on future games to predict past ones, which isn't how it actually gets used.

**Accuracy (CV MAE across 5 time-ordered folds):**

| Stat | Avg Error |
|---|---|
| Points | ±5.73 |
| Rebounds | ±2.15 |
| Assists | ±1.74 |
| 3PT Made | ±1.17 |
| PRA | ±7.03 |
| Steals | ±0.84 |
| Blocks | ±0.66 |
| Turnovers | ±1.13 |

Predictions get converted to probabilities by treating the model's error as a normal distribution — if the model predicts 27.3 points with a historical MAE of 5.73, we can calculate the probability that the actual value clears any given line.

---

## Daily retraining

GitHub Actions runs `train_model.py` every morning at 11am ET. It pulls last night's game logs, retrains all 10 models, and pushes the new `nba_models.pkl` back to the repo. Streamlit picks it up automatically. No manual work.

---

## Project structure

```
nba-props/
├── app.py              # Streamlit app — UI, live data fetching, predictions
├── train_model.py      # Data pipeline — nba_api pulls, feature engineering, XGBoost training
├── nba_models.pkl      # Trained models (updated daily by GitHub Actions)
├── requirements.txt
└── .github/
    └── workflows/
        └── retrain.yml # Daily retraining schedule
```

---

## Running locally

```bash
git clone https://github.com/ryanj06/nba-props.git
cd nba-props
pip install -r requirements.txt

# Train the model first — takes ~45 min because of nba_api rate limiting
python train_model.py

# Run the app
streamlit run app.py
```

---

## Limitations

- Doesn't account for injuries or lineup changes — always check injury reports
- Opponent defensive ratings are season-wide averages, not as-of-date
- Low confidence flag shown for players with fewer than 30 games
- Predictions assume normal minutes

---

*For entertainment purposes only. Bet responsibly.*
