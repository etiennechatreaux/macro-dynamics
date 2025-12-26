# Macro Dynamics - Regime Detection Pipeline

Pipeline de prétraitement pour la détection de régimes macroéconomiques.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Prétraitement avec recette par défaut (z_plus_momentum)
python -m macrostate.cli preprocess --input data/raw/raw_dataset.xlsx

# Choisir une recette spécifique
python -m macrostate.cli preprocess --recipe baseline_z

# Filtrer jusqu'à une date
python -m macrostate.cli preprocess --recipe z_plus_momentum --asof 2024-12-31

# Lister les recettes disponibles
python -m macrostate.cli list-recipes
```

## Recettes disponibles

| Recette | Description |
|---------|-------------|
| `baseline_z` | YC_SLOPE + z-scores rolling sur niveaux |
| `z_plus_momentum` | baseline_z + Δ1M/Δ6M + drawdown SPX |
| `changes_only` | Diffs/momentum uniquement, pas de z-scores |
| `levels_only` | Z-scores uniquement, pas de diffs |

## Prévention des fuites de données (Leakage)

Le transformer `RollingZScoreTransformer` utilise `shift(1)` avant de calculer la moyenne et l'écart-type rolling :

```python
shifted = X[col].shift(1)
rolling_mean = shifted.rolling(window=60).mean()
rolling_std = shifted.rolling(window=60).std()
z_score = (X[col] - rolling_mean) / rolling_std
```

Cela garantit que le z-score au temps t utilise uniquement les données de t-1 et antérieures.

## Structure

```
src/macrostate/
├── config/settings.py     # Configuration
├── data/io.py             # Chargement/sauvegarde
├── data/validation.py     # Validation des données
├── features/transformers.py  # Transformers sklearn
├── pipelines/preprocess.py   # Builder de pipelines
├── utils/                 # Logging, paths
└── cli.py                 # Interface CLI
```

## Outputs

- `data/cleaned/monthly.parquet` - Données nettoyées
- `data/features/<recipe>.parquet` - Features par recette
- `reports/data_quality.json` - Rapport qualité

