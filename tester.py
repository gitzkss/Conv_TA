# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from collections import defaultdict

import torch
import numpy as np
from torch import nn

from dataset import Dataset
from scripts import shredFacts

class Tester:
    def __init__(self, params,dataset, model_path,valid_or_test):
        self.model = torch.load(model_path)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.params = params
        self.model_path = model_path

        self.multi_tails_map = defaultdict(set)
        split_data = self.dataset.data['valid' if 'valid' in valid_or_test else 'test']
        for h, r, t, y, m, d in split_data:
            key = (int(h), int(r), int(y), int(m), int(d))
            self.multi_tails_map[key].add(int(t))

        self.keys_tensor = torch.tensor(
            list(self.multi_tails_map.keys()),
            dtype=torch.long
        ).cuda()

        self.values_tensor = torch.zeros(
            (len(self.multi_tails_map), self.dataset.numEnt()),
            device='cuda'
        )
        for idx, tails in enumerate(self.multi_tails_map.values()):
            self.values_tensor[idx, list(tails)] = 1.0


    def get_ranks(self, sim_scores, query_keys):
        matches = (self.keys_tensor.unsqueeze(1) == query_keys).all(dim=2).float()
        valid_mask = matches.any(dim=0)

        target_matrix = torch.zeros_like(sim_scores)
        if valid_mask.any():
            key_idx = matches[:, valid_mask].argmax(dim=0)
            target_matrix[valid_mask] = self.values_tensor[key_idx]

        sorted_idx = torch.argsort(sim_scores, dim=1, descending=True)
        sorted_targets = torch.gather(target_matrix, 1, sorted_idx)

        first_pos = (sorted_targets.cumsum(dim=1) > 0).int().argmax(dim=1)
        ranks = first_pos + 1

        ranks = ranks.float().masked_fill(~valid_mask, float('inf'))
        return ranks


    def test(self,is_save=True):
        all_ranks = []
        last_batch = False

        while not last_batch:
            heads, rels, _, years, months, days = self.dataset.next_valid_or_test_batch(
                self.params.bsize, self.valid_or_test)
            sim_scores = self.model(heads, rels, _, years, months, days)

            query_keys = torch.stack([heads, rels, years, months, days], dim=1)

            ranks = self.get_ranks(sim_scores, query_keys)
            all_ranks.append(ranks[ranks != float('inf')])

            last_batch = self.dataset.wasLastBatch()

        all_ranks = torch.cat(all_ranks)
        hits1 = (all_ranks <= 1).float().mean().item()
        hits3 = (all_ranks <= 3).float().mean().item()
        hits10 = (all_ranks <= 10).float().mean().item()
        mrr = (1.0 / all_ranks).mean().item()

        print(f"######{self.valid_or_test}######")
        print(f"hits@1 : {hits1}")
        print(f"hits@3 : {hits3}")
        print(f"hits@10 : {hits10}")
        print(f"MRR : {mrr}")
        print("\n")

        if is_save:
            self._save_results(hits1, hits3, hits10, mrr)

        return mrr

    def _save_results(self, hits1, hits3, hits10, mrr):
        model_name = self.params.model
        with open(f'result/{model_name}_{self.dataset.name}_{self.params.ne}_{self.params.bsize}_{self.params.lr}_{self.params.ker_sz}_{self.params.perm}.txt','a',encoding='utf-8') as file:
            file.write(f"######{self.valid_or_test}######\n")
            file.write(f"hits@1 : {hits1}\n")
            file.write(f"hits@3 : {hits3}\n")
            file.write(f"hits@10 : {hits10}\n")
            file.write(f"MRR : {mrr}\n\n")

        
