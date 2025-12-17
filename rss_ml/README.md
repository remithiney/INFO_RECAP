# rss-ml

Pipeline RSS : ingestion, NLP, clustering, lecture CLI.

## Installation

```bash
pip install -e .[dev]
# Optionnel : fastText pour détection de langue rapide
pip install .[fasttext]
# Optionnel : Textual pour un lecteur TUI
pip install .[textual]
```

Torch : choisissez la build adaptée (CPU/GPU) via https://pytorch.org/get-started/locally/.

## Commandes (MVP prévu)

- `rssml ingest --feeds data/feeds.txt` : charger les flux, récupérer articles, écrire `data/articles_raw.csv`.
- `rssml featurize` : langues, résumé, classification → `data/articles_features.csv`.
- `rssml cluster` : embeddings + clustering → mapping cluster.
- `rssml summarize-clusters` : résumés de cluster → `data/clusters.csv`.
- `rssml read-clusters data/clusters.csv` : lecture filtrable des clusters (pagination Rich).

## Données

- `data/feeds.txt` : liste de flux RSS.
- `data/articles_raw.csv` : articles bruts (ingestion).
- `data/articles_features.csv` : articles enrichis (NLP).
- `data/clusters.csv` : clusters résumés.***
