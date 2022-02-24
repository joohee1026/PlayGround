# FeedbackPrize

### Directory

    │
    ├── feedback-prize-2021
    │     ├── train
    │     │     ├── 73D6F19E24BD.txt
    │     │     ├── 62C57C524CD2.txt
    │     │     └── ...
    │     ├── test
    │     │     ├── D46BCB48440A.txt
    │     │     ├── D72CB1C11673.txt
    │     │     └── ...
    │     ├── train.csv
    │     └── train_prep_len.json
    │     
    ├── config.yaml
    ├── main.py
    ├── model.py
    ├── loader.py    
    ├── solver.py
    └── utils.py

Download [train_prep_len.json](https://drive.google.com/file/d/1olwI3-jZvpoE68uCK8Wo2tr_WZgb7LVz/view?usp=sharing)


### Usage
check for `config.yaml`

```
python main.py
```

```
config.yaml

    seed:                 2022 고정
    gpu_no:               사용할 gpu 번호
    device:               cuda 또는 cpu
    save_dir:             구성 및 학습 모델 저장 경로

    log
      logging:            wandb 내 학습 과정 기록 여부
      project:            wandb 프로젝트 이름
      entity:             wandb 사용자 이름
      name:               wandb 내 실행 이름
      group_name:         wandb 내 각 실행의 그룹 이름
      fold_cont:          모든 fold의 연속 기록 여부

    data
      data_dir:           사용 데이터 Directory
      max_length:         Tokenizer에 적용되는 최대 길이
      stride:             Overlapping token의 수, 0이면 overlap 없음

    train
      n_epoch:            학습 횟수
      n_fold:             사용할 fold의 수, 1이면 holdout setting
      n_batch:            1회 학습에 사용되는 데이터의 수
      amp:                Mixed precision training 사용 여부
      max_grad_norm:      Gradient clipping

    model
      name:               모델 이름 ex) bert-base-uncased

    optimizer
      type:               Optimizer 이름
      learning_rate:      학습률
      min_learning_rate:  Learning rate의 lower bound
      weight_decay:       Weight decay (L2 penalty)
      beta_1:             Adam optimizer 내 사용되는 coefficient
      beta_2:             Adam optimizer 내 사용되는 coefficient

    scheduler:
      type:               Scheduler 이름
      t_0:                CosineAnnealingWarmRestarts의 시작할 반복 수
      t_max:              CosineAnnealingLR의 최대 반복 수      

```
