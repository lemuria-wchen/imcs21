import json
import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples):
    ids = pad_sequence([sample['ids'] for sample in samples], padding_value=0)
    mask = pad_sequence([sample['mask'] for sample in samples], padding_value=0)
    token_type_ids = pad_sequence([sample['token_type_ids'] for sample in samples], padding_value=0)
    bio_ids = pad_sequence([sample['bio_ids'] for sample in samples], padding_value=0)
    chunks = [sample['chunks'] for sample in samples]
    bounds = [sample['bounds'] for sample in samples]
    return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'bio_ids': bio_ids,
            'chunks': chunks,
            'bounds': bounds
    }


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utts = self.data[index]['utts']
        bio_ids = self.data[index]['bio_ids']
        chunks = self.data[index]['chunks']
        bounds = self.data[index]['bounds']

        inputs = self.tokenizer.encode_plus(
            utts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_offsets_mapping=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        offset_mapping = inputs['offset_mapping']

        char2id = {}
        mapped_bio_ids = []
        for idx, om in enumerate(offset_mapping):
            if om is None:
                mapped_bio_ids.append(0)
            else:
                for offset in range(om[0], om[1]):
                    char2id[offset] = idx
                mapped_bio_ids.append(bio_ids[om[0]])
        assert len(ids) == len(mapped_bio_ids)

        mapped_chunks = []
        for chunk in chunks:
            if chunk[0] in char2id and chunk[1] in char2id:
                mapped_chunks.append([char2id.get(chunk[0]), char2id.get(chunk[1]), chunk[2], chunk[3]])

        mapped_bounds = []
        for bound in bounds:
            if bound in char2id:
                mapped_bounds.append(char2id.get(bound))
            else:
                mapped_bounds.append(len(ids) - 2)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'bio_ids': torch.tensor(mapped_bio_ids, dtype=torch.long),
            'chunks': mapped_chunks,
            'bounds': mapped_bounds
        }


def collate_fn_test(samples):
    ids = pad_sequence([sample['ids'] for sample in samples], padding_value=0)
    mask = pad_sequence([sample['mask'] for sample in samples], padding_value=0)
    token_type_ids = pad_sequence([sample['token_type_ids'] for sample in samples], padding_value=0)
    bounds = [sample['bounds'] for sample in samples]
    return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'bounds': bounds
    }


class CustomTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utts = self.data[index]['utts']
        bounds = self.data[index]['bounds']

        inputs = self.tokenizer.encode_plus(
            utts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_offsets_mapping=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        offset_mapping = inputs['offset_mapping']

        char2id = {}
        for idx, om in enumerate(offset_mapping):
            if om is not None:
                for offset in range(om[0], om[1]):
                    char2id[offset] = idx

        mapped_bounds = []
        for bound in bounds:
            if bound in char2id:
                mapped_bounds.append(char2id.get(bound))
            else:
                mapped_bounds.append(len(ids) - 2)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'bounds': mapped_bounds
        }


class BERTClass(torch.nn.Module):
    def __init__(self, model_name: str, enc_dim: int, num_bio: int, num_sn: int, num_sl: int):
        super(BERTClass, self).__init__()
        self.encoder = transformers.BertModel.from_pretrained(model_name)
        self.fc_bio = torch.nn.Linear(enc_dim, num_bio)
        self.fc_sn = torch.nn.Linear(enc_dim, num_sn)
        self.fc_sl = torch.nn.Linear(enc_dim, num_sl)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        return outputs[0]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str, indent=None):
    with open(path, 'w', encoding='utf-8') as f:
        if not indent:
            json.dump(data, f, ensure_ascii=False)
        else:
            json.dump(data, f, indent=indent, ensure_ascii=False)


def get_entity_bio(seq):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks
