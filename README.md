# MANTIS — Time Series Classification Foundation Model (From Scratch)

A from-scratch PyTorch implementation of **Mantis**, a lightweight, calibrated
foundation model for time series classification, based on the paper
*"Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series
Classification."* Every component — tokenization, attention, contrastive
pretraining, calibration — is implemented from raw PyTorch primitives, not a
pre-built library.

## What this implements

- **Dual-stream tokenizer** — raw values + first-differences, each processed
  through independent Conv1d + adaptive pooling, then fused into 32 patch tokens.
- **CLS token + sinusoidal positional encoding**, feeding into a 6-layer
  Transformer encoder (8-head attention, GELU, pre-LayerNorm).
- **Contrastive pretraining** (SimCLR-style NT-Xent loss) using a
  RandomCropResize augmentation, trained without any labels, with a
  warmup + cosine learning-rate schedule.
- **Frozen-encoder fine-tuning** — a linear classification head trained on
  labeled data on top of the frozen pretrained embeddings, to avoid
  overfitting a large encoder on a small labeled set.
- **Temperature scaling calibration** — a single learned scalar (fit via
  LBFGS on a held-out split) to make the model's confidence scores honest,
  matching the original paper's calibration focus.

## Validated results (UCR `GunPoint` dataset)

| Stage | Metric | Value |
|---|---|---|
| k-NN check on raw pretrained embeddings (no fine-tuning) | Test accuracy | 70.67% (vs 50% random baseline) |
| Fine-tuned linear classifier (single-dataset pretraining) | Test accuracy | 85.33% |
| Fine-tuned linear classifier (pooled multi-dataset pretraining, 627 series) | Test accuracy | **89.33%** |
| Calibration | Learned temperature | 0.81–1.18 (varies by validation split) |

Pretraining on a pooled set of 5 UCR datasets (GunPoint, Coffee, ECG200,
Wine, Beef — 627 series total, standardized to length 512) outperformed
single-dataset pretraining, supporting the paper's premise that broader
unsupervised pretraining produces better general-purpose embeddings.

**On calibration:** temperature scaling correctly identifies and corrects
over/under-confidence, but the effect size varies across runs — likely due
to the small (~75-example) held-out calibration split available from this
dataset's test set. A larger calibration set would give a more stable
estimate.

## Exploratory extension: real market data

As a stress test beyond benchmark data, the same architecture was applied
to real daily stock price data (window = 60 trading days, predicting
5-day-ahead direction). **Honest result: price history alone did not
produce a reliable directional signal** — accuracy hovered around the
naive majority-class baseline (~53%), consistent with the efficient-market
expectation that simple price-shape patterns are not a reliable trading
edge on their own. This is treated here as a legitimate negative result,
not a shortcoming to hide — real predictive signal in markets typically
requires richer inputs (volume, fundamentals, order flow) well beyond
price alone.

## Project structure
tokenizer.py # dual-stream tokenizer (raw + diff → 32 patch tokens)
encoder.py # CLS token + positional encoding + Transformer encoder
model.py # full MantisModel + ProjectionHead
augment.py # RandomCropResize augmentation
loss.py # NT-Xent contrastive loss
dataset.py # UCR/UEA dataset loader + multi-dataset pooling (via aeon)
classifier.py # pretraining + fine-tuning + checkpoint saving
calibrate.py # temperature scaling calibration
stock_data.py # real stock price windowing/labeling (exploratory)
stock_pool.py # multi-ticker unlabeled pretraining pool (exploratory)
stock_predict.py # stock direction fine-tuning + evaluation (exploratory)

## Running it

```bash
pip install torch aeon yfinance

python classifier.py   # pretrains + fine-tunes + saves trained_mantis.pt
python calibrate.py    # loads the trained model, fits calibration
```

## What's simplified vs. the original paper

- Two input streams (raw + diff) instead of the paper's four (raw, diff,
  segment mean, segment std) — valid given z-normalized inputs, per the
  paper's own note that normalized inputs make the extra streams less
  informative.
- Pretraining pool is 5 UCR datasets (~20K windows in the stock variant),
  smaller than the paper's large-scale pretraining corpus.
- Univariate only — the multivariate channel adapters described in the
  paper aren't implemented here.

## Next steps

- Pool more UCR datasets for pretraining diversity.
- Multivariate channel adapters (PCA/SVD/random-projection/learned-linear).
- If pursuing the market-data direction further: richer input features
  (volume, cross-asset signals), longer/shorter prediction horizons, and
  proper walk-forward validation across many tickers.
