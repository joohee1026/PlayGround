import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import torch.optim as optim


def confirm_params(cfg):
    assert isinstance(cfg.seed, int)
    assert isinstance(cfg.gpu_no, int)
    assert cfg.device in ["cuda", "cpu"]
    assert cfg.train.n_fold in [1, 3, 5, 10]


def fix_seed(seed=2022):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_split_info(data_path=None, save_path=None, seed=2022):
    if data_path is None:
        data_path = "./feedback-prize-2021/train_prep.json"
    data = pd.read_json(data_path)

    fix_seed(seed)

    data["total_length"] = [len(c) for c in data["text_split"]]
    n_bins = int(np.floor(1 + np.log2(len(data))))
    data["bins"] = pd.cut(data["total_length"], bins=n_bins, labels=False)
    bins = data.bins.to_numpy()

    for k in [3, 5, 10]:
        data[f"f{k}"] = 0
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for f, (_, valid_idx) in enumerate(kfold.split(X=data, y=bins)):
            data.loc[valid_idx, f"f{k}"] = f

    if save_path is not None:
        data.to_json(os.path.join(save_path, "train_prep_len.json"))
    else:
        return data


def prep(df, data_dir, save_path=None):
    ids, texts = [], []
    for f in tqdm(list(os.listdir(data_dir))):
        ids.append(f.replace(".txt", ""))
        texts.append(open(os.path.join(data_dir, f), "rb").read())
        data = pd.DataFrame({"id": ids, "text": texts})
    data["text_split"] = data.text.str.split()

    entities = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        e = ["O"] * len(row["text_split"])
        for _, r in df[df["id"] == row["id"]].iterrows():
            discourse = r["discourse_type"]
            ps = [int(x) for x in r["predictstring"].split(" ")]
            e[ps[0]] = f"B-{discourse}"
            for k in ps[1:]:
                e[k] = f"I-{discourse}"
        entities.append(e)
    data["entities"] = entities

    if save_path is not None:
        data.to_json(os.path.join(save_path, "train_prep.json"))
    else:
        return data


def holdout(data, aids, train_ratio):
    train_idx = np.random.choice(np.arange(len(aids)), int(train_ratio*len(aids)), replace=False)
    valid_idx = np.setdiff1d(np.arange(len(aids)), train_idx)

    data_train = data.loc[data["id"].isin(aids[train_idx])].reset_index(drop=True)
    data_valid = data.loc[data["id"].isin(aids[valid_idx])].reset_index(drop=True)
    return data_train, data_valid


def calc_overlap(row):
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))

    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    gt_df = gt_df[['id','discourse_type','predictionstring']].reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']].reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index

    joined = pred_df.merge(gt_df,
                           left_on=['id','class'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score


def set_optimizer(opt_config, model):
    opt_name = opt_config.type.lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=opt_config.learning_rate,
            betas=(opt_config.beta_1, opt_config.beta_2),
            weight_decay=opt_config.weight_decay
        )
    elif opt_name == "adamw":
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=opt_config.learning_rate,
            betas=(opt_config.beta_1, opt_config.beta_2),
            weight_decay=opt_config.weight_decay
        )
    elif opt_name == "radam":
        optimizer = optim.RAdam(
            params=model.parameters(),
            lr=opt_config.learning_rate,
            betas=(opt_config.beta_1, opt_config.beta_2),
            weight_decay=opt_config.weight_decay
        )
    else:
        raise ValueError
    return optimizer


def set_scheduler(sch_config, opt_config, optimizer):
    sch_name = sch_config.type.lower()
    if sch_name == "CosineAnnealingLR".lower():
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=sch_config.t_max,
            eta_min=opt_config.min_learning_rate
        )
    elif sch_name == "CosineAnnealingWarmRestarts".lower():
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=sch_config.t_0,
            eta_min=opt_config.min_learning_rate
        )
    elif sch_name == "ReduceLROnPlateau".lower():
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=sch_config.factor,
            patience=sch_config.patience,
            min_lr=opt_config.min_learning_rate
        )
    elif sch_name == "LambdaLR".lower():
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda= lambda epoch: 0.9*epoch
        )
    elif sch_name == "LinearLR".lower():
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=sch_config.start_factor,
            total_iters=sch_config.total_iters
        )
    else:
        scheduler = None
    return scheduler
