from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
import ast
from torchvision.transforms import transforms
import os

from PIL import Image
import cv2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
img_trans = img_aug = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])


def preprocess_frame(video, max_frames):
    num_frames = video.shape[0]
    dim = video.shape[1]
    padding_length = max_frames - num_frames
    cls_num_frames = num_frames
    mask = [1] * cls_num_frames + ([0] * padding_length)
    fillarray = np.zeros((padding_length, dim))
    video_out = np.concatenate((video, fillarray), axis=0)
    video_out = torch.tensor(video_out, dtype=torch.float32).cuda()
    mask = torch.tensor(np.array(mask)).cuda()
    return video_out, mask


        


class MuliDataset(Dataset):

    def __init__(self, df_, device, transform=None):
        self.data_path = df_
        self.device = device

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = self.data_path.iloc[idx, 0].split('/')[-1].split('.')[0] + '.mp4'
        
        video_ = np.zeros((1, 1024))
        if os.path.exists(self.data_path.iloc[idx, 0]):
            video_ = np.load(self.data_path.iloc[idx, 0])

            
        video_, video_mask = preprocess_frame(video_, 120)
        
        video_ = [video_, video_mask]
        
        audio_ = np.zeros((1, 128))
        if os.path.exists(self.data_path.iloc[idx, 1]):
            audio_ = np.load(self.data_path.iloc[idx, 1])
        audio_, audio_mask = preprocess_frame(audio_, 120)
        audio_ = audio_[:, :128]
        audio_ = [audio_, audio_mask]
#         print(audio_.shape)

        # token_ids, seq_len, mask
        text_ = [(torch.LongTensor(iter_[0]).to(self.device), torch.tensor(int(iter_[1])).to(self.device), torch.tensor(np.array(iter_[2])).to(self.device)) for iter_ in ast.literal_eval(self.data_path.iloc[idx, 3])]

#         label_index = [int(iter_) for iter_ in ast.literal_eval(self.data_path.iloc[idx, 4])]
#         label_ = torch.zeros(82).to(self.device)
#         label_[label_index] = 1.


        return file_name, text_, video_, audio_


