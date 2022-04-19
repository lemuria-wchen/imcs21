import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from collections import defaultdict

from utils import CustomTestDataset, collate_fn_test, load_json, write_json, get_entity_bio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cuda_num", default='0', required=True, type=str)
args = parser.parse_args()


prefix = 'data'
model_prefix = 'saved'

# load test set
test = load_json(os.path.join(prefix, 'test.json'))

# load symptoms
id2sym = {key: value for key, value in pd.read_csv('../../../dataset/symptom_norm.csv')['norm'].items()}
id2bio = {idx: item for idx, item in enumerate(['PAD', 'O', 'B-Symptom', 'I-Symptom'])}
id2sl = {0: '0', 1: '1', 2: '2'}


# settings
device = 'cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu'

model_name = 'bert-base-chinese'

TEST_BATCH_SIZE = 32
MAX_LEN = 256

# data loader
tokenizer = BertTokenizerFast.from_pretrained(model_name)
test_set = CustomTestDataset(test, tokenizer, MAX_LEN)

test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
test_loader = DataLoader(test_set, collate_fn=collate_fn_test, **test_params)

# load model
model = torch.load(os.path.join(model_prefix, 'model.pkl'))
model.to(device)

# inference
preds = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        outputs = model(batch['ids'].to(device), batch['mask'].to(device), batch['token_type_ids'].to(device))
        bio_outputs = model.fc_bio(outputs)
        bio_tag_ids = torch.argmax(bio_outputs, dim=-1)
        bounds = batch['bounds']
        for i in range(len(bounds)):
            bound = bounds[i]
            bio_tag_id = bio_tag_ids[:, i]
            bio_tg_span = [id2bio.get(tag.item()) for tag in bio_tag_id[bound[0]: bound[1] + 1]]
            chunks = get_entity_bio(bio_tg_span)
            if len(chunks) > 0:
                features = []
                for chunk in chunks:
                    features.append(torch.mean(outputs[bound[0] + chunk[1]: bound[0] + chunk[2] + 1, i, :], dim=0))
                sn_outputs = model.fc_sn(torch.stack(features))
                sl_outputs = model.fc_sl(torch.stack(features))
                sn_ids = torch.argmax(sn_outputs, dim=-1).tolist()
                sl_ids = torch.argmax(sl_outputs, dim=-1).tolist()
                pred = {id2sym.get(sn_id): id2sl.get(sl_id) for sn_id, sl_id in zip(sn_ids, sl_ids)}
            else:
                pred = {}
            # print(pred)
            preds.append(pred)


pids = [sample['pid'] for sample in test]

final = defaultdict(dict)
for pid, pred in zip(pids, preds):
    final[pid].update(pred)

write_json(final, path='imp_pred.json', indent=4)
