import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import math

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, is_phase=False, min_db=-80, max_db=0):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.is_phase = is_phase
        self.min_db = min_db
        self.max_db = max_db

        # 파일명에서 condition 추출
        self.conditions = [os.path.basename(f).split("_")[0] for f in self.files]
        unique_conditions = sorted(set(self.conditions))
        self.cond2idx = {c: i for i, c in enumerate(unique_conditions)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])  # (129,376)
        '''
        if self.is_phase:
            # phase ∈ [-π, π]
            phase = spec
            spec_norm = np.stack([np.sin(phase), np.cos(phase)], axis=0)  # (2,H,W)
        else:
            spec_db = 20*np.log10(spec + 1e-8)
            spec_db = np.clip(spec_db, self.min_db, self.max_db)
            spec_norm = (spec_db - self.min_db) / (self.max_db - self.min_db) * 2 - 1
            spec_norm = np.expand_dims(spec_norm, axis=0)

        spec_norm = torch.tensor(spec_norm, dtype=torch.float32)
        '''
        spec = torch.tensor(spec, dtype=torch.float32)
        cond_idx = self.cond2idx[self.conditions[idx]]  # int label
        return spec, cond_idx