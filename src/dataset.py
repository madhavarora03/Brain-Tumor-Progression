import json
import os

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BraTSPRODataset(Dataset):
    def __init__(self, root_dir, patients_json, use_registered=True, use_seg=False, transform=None):
        self.root_dir = root_dir
        self.use_registered = use_registered
        self.use_seg = use_seg
        self.transform = transform

        with open(patients_json, 'r') as f:
            self.data = json.load(f)

        self.samples = []
        for patient_id, cases in self.data.items():
            for case_id, case_data in cases.items():
                if "response" in case_data:  # Ignore test set
                    self.samples.append((patient_id, case_id, case_data))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, case_id, case_data = self.samples[idx]

        prefix = "baseline_registered" if self.use_registered else "baseline"
        suffix = "followup_registered" if self.use_registered else "followup"

        baseline_path = os.path.join(self.root_dir, case_data[prefix])
        followup_path = os.path.join(self.root_dir, case_data[suffix])

        baseline_vol = self._load_mri_modalities(baseline_path)
        followup_vol = self._load_mri_modalities(followup_path)

        x = np.concatenate([baseline_vol, followup_vol], axis=0)  # Shape: (8, H, W, D)
        x = torch.tensor(x, dtype=torch.float32)

        y = case_data['response']
        y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y

    @staticmethod
    def _load_mri_modalities(folder):
        channels = []
        for i in range(4):  # T1, T1c, T2, FLAIR
            nii_path = os.path.join(folder, f"{os.path.basename(folder)}_000{i}.nii.gz")
            vol = nib.load(nii_path).get_fdata()
            vol = vol.astype(np.float32)
            vol = (vol - vol.mean()) / (vol.std() + 1e-8)  # Normalize
            channels.append(vol[np.newaxis])
        return np.concatenate(channels, axis=0)


def test():
    dataset = BraTSPRODataset(root_dir="data", patients_json="data/patients.json")
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(next(iter(dataloader))[0].shape)

if __name__ == "__main__":
    test()
