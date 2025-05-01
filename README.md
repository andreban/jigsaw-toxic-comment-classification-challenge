# Kaggle - Toxic Comment Classification Challenge

This repository implements a classification solution for Kaggle's [Toxic Comment Classification Challenge][1].

## Dataset preparation

This project only uses the `train.csv` from the origina dataset, which should be copied to `.data/train.csv`.
Embeds for the training set are then generated with `generate_embeds_dataset.py`.

## Project Setup
This project uses [Poetry][2] to manage dependencies. Make sure it's installed.

```shell
python3 -m venv .venv
source .venv/bin/activate
poetry install
```

[1]: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/
[2]: https://python-poetry.org/