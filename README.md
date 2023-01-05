# Korean-Language-Classification
> 한국어 문장 유형 분류 모델 

## Description
KoBERT 분류 모델을 활용한 한국어 문장 유형 분류에서 최적의 하이퍼 파라미터 튜닝을 위해 wandb의 sweep을 사용한 프로젝트

## Environments
- Google Colab
- GPU

## Requirements
- wandb==0.13.7 
- transformers==4.25.1

## Tune Method
- Random search

## Metric
- Minimize loss

## Hyper-parameters
- epochs -> [4,5,6,7,8]
- learning rate -> [1e-5,2e-5,1e-3]
- batch size -> [32, 64]
- dropout rate -> [0.1,0.2,0.3]
- optimizer -> [Adam, AdamW, AdamP]
- scheduler -> [linear, cosine]
- warmup_steps -> [0, 100]

## Data
- 한국어 문장 분류 데이터셋 약 16,000 문장 
  (Dacon, 성균관대학교, 문장 유형 분류 AI 경진대회, https://dacon.io/competitions/official/236037/data)

## Model
- KoBERT (70GB Korean text dataset and 42000 lower-cased subwords are used)
- reference : https://github.com/kiyoungkim1/LMkor

## run
- [run this notebook](https://github.com/nimnuyh/Korean-Language-Classification/blob/119b683fc9a2b46a61f8714c0298e8685ddff4a3/run.ipynb)

## add
- main.py의 config로 hyper parameter 조정
