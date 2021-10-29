import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np
import PIL
import torchvision.transforms as transforms
import os
import pandas as pd

from preprocessing import center_crop_and_resize

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


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
        
        if height > width:
            image_size = width
        else:
            image_size = height
            
        if image_size >= 512:
            f_size = 512
        elif image_size >= 448:
            f_size = 448
        elif image_size >= 336:
            f_size = 336
        else:
            f_size = 224
        
        trans = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.Resize(f_size, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
        
        frames = torch.FloatTensor(num_frames, channels, f_size, f_size)

        # load the video to tensor
        for idx in range(num_frames):
            frame = Image.open(frames_path[idx])
            frame = trans(frame)
            frames[idx, :, :, :] = frame
            
        return {'video': frames, 'input': video_path, 'output': output_file}


    