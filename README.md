# Trade Bot Colab Notebooks

This repository contains standalone ML notebooks for the trade-bot project.

## Files
- `regime_classifier_training.ipynb` — End-to-end regime classifier training notebook
- `.env.example` — Credentials template for Massive S3 access
- `register_model.py` — Register trained models to MLflow
- `train_regime.py` — Script-based training alternative

## Open in Google Colab
1. Go to https://colab.research.google.com/
2. File -> Open notebook -> GitHub
3. Select this repository and open `regime_classifier_training.ipynb`

## Credentials
Use one of:
- Colab Secrets: `MASSIVE_ACCESS_KEY` and `MASSIVE_SECRET_KEY`
- Env vars: `MASSIVE_S3_ACCESS_KEY` and `MASSIVE_S3_SECRET_KEY`
- `.env` file copied from `.env.example`

Never commit real credentials.
