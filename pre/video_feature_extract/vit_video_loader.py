import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np
import PIL
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from preprocessing import center_crop_and_resize

model = timm.create_model('vit_large_patch32_384', pretrained=False)

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
model = model.cuda()
model_data = torch.load('pretrain_models/vit-large/jx_vit_large_p32_384-9b920ba8.pth')
model.load_state_dict(model_data)

def get_model():
    model.eval()
    print('loaded')
    return model


class VideoLoader(Dataset):
    def __init__(self, csv_file,image_size):
        
        
        self.image_size = image_size
        self.csv = pd.read_csv(csv_file)
    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.csv)
    def __getitem__(self, index):
        """ get a video """
        video_path = self.csv['video_path'].values[index]
        output_file = self.csv['feature_path'].values[index]
        video_path = self.csv.iloc[index].video_path
        
        frames_path = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))])
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)
    
    
        
        frames = torch.FloatTensor(num_frames, channels, 384, 384)

        # load the video to tensor
        for idx in range(num_frames):
            frame = Image.open(frames_path[idx]).convert('RGB')
            frame = transform(frame).unsqueeze(0)
            frames[idx, :, :, :] = frame
            
        return {'video': frames, 'input': video_path, 'output': output_file}


    