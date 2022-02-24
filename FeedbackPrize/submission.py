import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

MODEL_PATH = _ ### YOUR MODEL PATH
MODEL_CHECKPOINT = _ ### YOUR MODEL

BATCH_SIZE = 4
DEVICE = 'cuda'
MAX_LENGTH = 1024
DOC_STRIDE = 0


def load_df_test():
    test_names, df_test = [], []
    for f in list(os.listdir('../input/feedback-prize-2021/test')):
        test_names.append(f.replace('.txt', ''))
        df_test.append(open('../input/feedback-prize-2021/test/' + f, 'r').read())
    df_test = pd.DataFrame({'id': test_names, 'text': df_test})
    df_test['text_split'] = df_test.text.str.split()
    return df_test


def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
    return label_ids


def tokenize(df):
    encoded = tokenizer(
        df['text_split'].tolist(),
        is_split_into_words=True,
        return_overflowing_tokens=True,
        stride=DOC_STRIDE,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True
    )
    encoded['word_ids'] = []
    n = len(encoded['overflow_to_sample_mapping'])
    for i in tqdm(range(n), total=n):
        text_idx = encoded['overflow_to_sample_mapping'][i]
        word_ids = encoded.word_ids(i)
        encoded['word_ids'].append([w if w is not None else -1 for w in word_ids])
    encoded = {key: torch.as_tensor(val) for key, val in encoded.items()}
    return encoded


class FBPDataset(Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {}
        for k in self.data.keys():
            item[k] = self.data[k][index]
        return item

    def __len__(self):
        return len(self.data['input_ids'])


def load_model():
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT))
    model.eval()
    print('Model loaded.')
    return model


df_test = load_df_test()

output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

map_clip = {'Lead':9, 'Position':5, 'Evidence':14, 'Claim':3, 'Concluding Statement':11,'Counterclaim':6, 'Rebuttal':4}

LABELS_TO_IDS = {v:k for k,v in enumerate(output_labels)}
IDS_TO_LABELS = {k:v for k,v in enumerate(output_labels)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

tokenized_test = tokenize(df_test)
n_tokens = len(tokenizer(df_test.iloc[2]['text'])['input_ids'])

ds_test = FBPDataset(tokenized_test)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = load_model()


def inference(dl):
    predictions = defaultdict(list)
    seen_words_idx = defaultdict(list)

    for batch in dl:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        outputs = model(ids, attention_mask=mask, return_dict=False)
        del ids, mask

        batch_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()
        for k, (chunk_preds, text_id) in enumerate(zip(batch_preds, batch['overflow_to_sample_mapping'].tolist())):
            word_ids = batch['word_ids'][k].numpy()
            chunk_preds = [IDS_TO_LABELS[i] for i in chunk_preds]
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx not in seen_words_idx[text_id]:
                    predictions[text_id].append(chunk_preds[idx])
                    seen_words_idx[text_id].append(word_idx)

    final_predictions = [predictions[k] for k in sorted(predictions.keys())]
    return final_predictions


def get_predictions(df, dl):
    all_labels = inference(dl)
    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_labels[i]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': pass
            else: cls = cls.replace('B','I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls != '' and end - j > 7:
                final_preds.append((idx, cls.replace('I-',''), ' '.join(map(str, list(range(j, end))))))
            j = end

    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id','class','predictionstring']
    return df_pred


def threshold(df):
    df = df.copy()
    df['len'] = df['predictionstring'].apply(lambda x:len(x.split()))
    for key, value in map_clip.items():
        index = df.loc[df['class']==key].query(f'len<{value}').index
        df.drop(index, inplace = True)
    df = df.drop('len', axis=1)
    return df


def post_process(df_sub):
    df_post = [df_sub.iloc[0].copy()]
    for i in range(1, len(df_sub)):
        prev_row = df_post[-1]
        row = df_sub.iloc[i].copy()

        if row['id'] == prev_row['id'] and row['class'] == prev_row['class']:
            try:
                first_pos_row = int(row['predictionstring'].split(" ")[0])
                last_pos_prev_row = int(prev_row['predictionstring'].split(" ")[-1])

                if last_pos_prev_row + 2 == first_pos_row:
                    new_id = last_pos_prev_row + 1
                    row['predictionstring'] = prev_row['predictionstring'] + f' {new_id} ' + row['predictionstring']
                    df_post = df_post[:-1]

                df_post.append(row)
            except:
                df_post.append(row)
        else:
            df_post.append(row)
    df_post = pd.DataFrame(df_post).reset_index(drop=True)
    df_post = threshold(df_post)
    return df_post


def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end]])


def link_evidence(oof):
    thresh = 1
    idu = oof['id'].unique()
    idc = idu[1]
    eoof = oof[oof['class'] == "Evidence"]
    neoof = oof[oof['class'] != "Evidence"]
    for thresh2 in range(26,27, 1):
        retval = []
        for idv in idu:
            for c in  ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement',
                   'Counterclaim', 'Rebuttal']:
                q = eoof[(eoof['id'] == idv) & (eoof['class'] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i,r in q.iterrows():
                    pst = pst +[-1] + [int(x) for x in r['predictionstring'].split()]
                start = 1
                end = 1
                for i in range(2,len(pst)):
                    cur = pst[i]
                    end = i
                    if (cur == -1 and c != 'Evidence') or ((cur == -1) and ((pst[i+1] > pst[end-1] + thresh) or (pst[i+1] - pst[start] > thresh2))):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end+1))
                retval.append(v)
        roof = pd.DataFrame(retval, columns = ['id', 'class', 'predictionstring'])
        roof = roof.merge(neoof, how='outer')
        return roof


df_sub = get_predictions(df_test, dl_test)
df_sub.head()

#df_post = post_process(df_sub)
df_post = threshold(df_sub)
df_post = link_evidence(df_post)
df_post.to_csv("submission.csv", index=False)
df_post
