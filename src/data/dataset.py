
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class RSNAPneumoniaDataset(Dataset):
    """
    Loads RSNA chest X-ray DICOMs + binary pneumonia labels.
    split_df: dataframe with columns [\'patientId\', \'Target\']
    """
    def __init__(self, split_df, image_dir, image_size=224, augment=False):
        self.df = split_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["patientId"]
        target = row["Target"]

        dcm_path = f"{self.image_dir}/{patient_id}.dcm"
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.stack([img] * 3, axis=-1)

        if self.augment:
            img = self._augment(img)

        img = img / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        target = torch.tensor(target, dtype=torch.float32)

        return img, target

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()

        if np.random.rand() < 0.5:
            angle = np.random.uniform(-8, 8)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img = np.clip(img * alpha + beta, 0, 255)

        return img
