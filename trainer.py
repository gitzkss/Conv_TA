import numpy as np

from dataset import Dataset
from params import Params
from tester import Tester
from Conv_TA import Conv_TA
import torch
import torch.nn as nn
import os
import time
from collections import defaultdict

class Trainer:
    def __init__(self, dataset, params, model_name, model_path="no"):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        if model_path == "no":
            self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params),
                                         device_ids=[0], output_device=0)
        else:
            self.model = torch.load(model_path)
        self.model = self.model.cuda()
        self.st_epoch = 1
        self.dataset = dataset
        self.params = params

        self.multi_tails_map = defaultdict(set)
        for h, r, t, y,m,d in self.dataset.data['train']:
            time_key = (h,r,y,m,d)
            self.multi_tails_map[time_key].add(int(t))
        self.keys_tensor = torch.tensor(
            list(self.multi_tails_map.keys()),
            dtype=torch.long
        ).cuda()  # shape: [num_keys, 5]

        self.values_tensor = torch.zeros(
            (len(self.multi_tails_map), self.dataset.numEnt()),
            device='cuda'
        )
        for idx, tails in enumerate(self.multi_tails_map.values()):
            self.values_tensor[idx, list(tails)] = 1.0

    def train(self, early_stop=False):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )

        params = list(self.model.parameters())
        k = sum(p.numel() for p in params)

        loss_f = nn.BCELoss()

        for epoch in range(1, self.params.ne + 1):
            total_loss = 0.0
            start = time.time()
            last_batch = False

            while not last_batch:
                optimizer.zero_grad()
                heads, rels, _, years, months, days = self.dataset.nextBatch(self.params.bsize, neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()
                scores = self.model(heads, rels, _, years, months, days)  # [batch, num_ent]
                query_keys = torch.stack([heads, rels, years, months, days], dim=1)
                targets = torch.zeros_like(scores)
                matches = (self.keys_tensor.unsqueeze(1) == query_keys).all(dim=2).float()
                row_idx = matches.any(dim=0).nonzero().squeeze(1)
                if row_idx.numel() > 0:
                    key_idx = matches[:, row_idx].argmax(dim=0)
                    targets[row_idx] = self.values_tensor[key_idx]
                smooth_targets = (1.0 - self.params.label_smoothing) * targets + \
                                 self.params.label_smoothing / targets.size(1)

                loss = loss_f(scores, smooth_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch} Loss: {total_loss:.4f} | Time: {time.time()-start:.1f}s")
            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

    def saveModel(self, chkpnt):
        directory = f"models/{self.model_name}/{self.dataset.name}/"
        os.makedirs(directory, exist_ok=True)
        model_path = f"{directory}{self.params.str_()}_{chkpnt}.chkpnt"
        torch.save(self.model, model_path)
        tester = Tester(self.params, self.dataset, model_path, "valid")
        tester.test(False)