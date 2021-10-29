import torch as th
import math
import numpy as np
from new_video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--csv',
    type=str,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='2d',
                            help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,
                            help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                            help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                            help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                            help='Resnext model path')
args = parser.parse_args()

dataset = VideoLoader(
    args.csv,
    image_size = 224
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
model = get_model(args)

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data['input'][0]
        output_file = data['output'][0]
        if len(data['video'].shape) > 3:
            print('Computing features of video {}/{}: {}'.format(
                k + 1, n_dataset, input_file))
            video = data['video'].squeeze()
            if len(video.shape) == 4:
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, 2560).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    batch_features = model.extract_features(video_batch)
                    batch_features = th.mean(batch_features, dim=[-2, -1])
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                np.save(output_file, features)
        else:
            print('Video {} already processed.'.format(input_file))
