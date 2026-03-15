# Trade Bot Colab Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KhangLy2606/trade-bot-colab-notebooks/blob/master/regime_classifier_training.ipynb)

This repository contains standalone ML notebooks for the trade-bot project.

## Files
- `regime_classifier_training.ipynb` — End-to-end regime classifier training notebook
- `.env.example` — Credentials template for Massive S3 access
- `register_model.py` — Register trained models to MLflow
- `train_regime.py` — Script-based training alternative

## Open in Google Colab
- Direct notebook link: https://colab.research.google.com/github/KhangLy2606/trade-bot-colab-notebooks/blob/master/regime_classifier_training.ipynb
- Or open Colab -> File -> Open notebook -> GitHub and search `KhangLy2606/trade-bot-colab-notebooks`

## Credentials
Use one of:
- Colab Secrets: `MASSIVE_ACCESS_KEY` and `MASSIVE_SECRET_KEY`
- Env vars: `MASSIVE_S3_ACCESS_KEY` and `MASSIVE_S3_SECRET_KEY`
- `.env` file copied from `.env.example`

Never commit real credentials.
