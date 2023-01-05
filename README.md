# Korean-Language-Classification
한국어 문장 분류 모델 

## Description
wandb의 sweep 기능을 이용해 최적의 하이퍼 파라미터를 찾아주는 한국어 문장 유형 분류 모델 

## requirements
- wandb==0.13.7 
- transformers==4.25.1

## tune method
- random search

## metric
- minimize loss

## hyper parameters
- epochs -> []
- learning rate -> []
- batch size -> []
- dropout rate -> []
- optimizer -> [Adam, AdamW, AdamP]
- scheduler -> [linear, cosine]
- warmup_steps -> []

## data
- 한국어 문장 분류 데이터셋 약 16,000 문장

## model
- kobert (70GB Korean text dataset and 42000 lower-cased subwords are used)
- reference : https://github.com/kiyoungkim1/LMkor

## 
