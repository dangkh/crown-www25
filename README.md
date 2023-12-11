# [WWW'24] CIDER: Category-aware Intent Disentanglement for Accurate News Recommendations

## The overview of KHAN
![The overview of KHAN](./assets/overview.PNG)

## Available dataset
1. [MIND Dataset](https://msnews.github.io/)
2. [Adressa Dataset](https://reclab.idi.ntnu.no/dataset/)

## Datasets
|Datasets|# of Users|# of News|Avg. title len|Avg. body len|
|:---:|:---:|:---:|:---:|:---:|
|MIND|94,057|65,238|11.67|41.01|
|Adressa|601,215|73,844|6.63|552.15|

## Dependencies
Our code runs on the Intel i7-9700k CPU with 64GB memory and NVIDIA RTX 2080 Ti GPU with 12GB, with the following packages installed:
```
python 3.8.10
torch 1.11.0
torchtext 0.12.0
pandas
numpy
argparse
sklearn
```
## How to run
```
python main.py --news_encoder=CIDER --user_encoder=CIDER
```

## Experiments Results for Rebuttal

 1. Recommendation accuracy of CIDER with/without the category information in MIND and Adressa datasets.

|                    | Adressa                      |||   | MIND-small                       |||  |
|--------------------|:-------:|:-----:|:------:|:-------:|:----------:|:-----:|:------:|:-------:|
|                    |   AUC   |  MRR  | nDCG@5 | nDCG@10 |     AUC    |  MRR  | nDCG@5 | nDCG@10 |
| CIDER w/o category |  83.85  | 52.84 |  55.33 |  60.35  |    68.06   | 32.82 |  36.51 |  42.59  |
|  CIDER w/ category |  84.13  | 53.93 |  56.64 |  61.15  |    68.23   | 33.54 |  36.97 |  42.93  |
