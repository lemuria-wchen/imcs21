import json
from collections import Counter


def load_data(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data


class Stats:
    def __init__(self, samples):
        self.samples = samples

    def num_samples(self):
        print('number of samples: {}'.format(len(self.samples)))

    def dis_stats(self):
        # stats of diseases
        dis = []
        for pid, sample in self.samples.items():
            dis.append(sample['diagnosis'])
        print('number of diseases: {}'.format(len(set(dis))))
        print('distribution of diseases: {}'.format(Counter(dis)))

    def sx_stats(self):
        # stats of symptoms
        sxs = []
        for pid, sample in self.samples.items():
            for sx in sample['explicit_info']['Symptom']:
                sxs.append(sx)
            for sx in sample['implicit_info']['Symptom']:
                sxs.append(sx)
        print('number of symptoms: {}'.format(len(set(sxs))))
        print('distribution of symptoms: {}'.format(Counter(sxs)))
