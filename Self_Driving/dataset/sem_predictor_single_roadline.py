import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import random

class Sem_predictor_single(Dataset):
    def __init__(self, raw_path, all_sem_cls, stage='train'):
        print("preparing dataset")
        self.data = []
        self.all_directions = ['left', 'right']
        # self.all_sem_classes = [6]
        self.all_sem_classes = all_sem_cls

        # towns = os.listdir(raw_path)

        if stage == 'train':
            towns = ['Town01']
        else:
            towns = ['Town02']

        # towns = ['Town02']

        for town in towns:
            episodes = os.listdir(osp.join(raw_path, town))
            random.shuffle(episodes)

            for episode in episodes:
                agents = os.listdir(osp.join(raw_path, town, episode))
                for agent in agents:
                    frames = os.listdir(osp.join(raw_path, town, episode, agent))
                    for frame_idx in range(len(frames)):
                        for direction in self.all_directions:
                            self.data.append((osp.join(raw_path, town, episode, agent, frames[frame_idx]), direction))

        print('dataset ready')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, j):

        folder, direction = self.data[j] # sample: list of length 10, every entry is one frame

        images = {}
        labels = {}
        for sem_id in self.all_sem_classes:
            images[sem_id] = None
            labels[sem_id] = None
        
        image = Image.open(osp.join(folder, direction + '.png')).convert('RGB')
        image, _, _ = image.split()
        image = np.array(image)
        image = (np.arange(13) == image[..., None])
        for sem_id in self.all_sem_classes:
            image_sem = image[:, :, sem_id:sem_id+1]
            image_sem = torch.tensor(image_sem, dtype=torch.float32)
            image_sem = image_sem.permute(2, 0, 1)
            images[sem_id] = image_sem

        with open(osp.join(folder,'aux_states.json'), 'r') as f:
            labels_dict = json.load(f)
            f.close()
        
        for sem_id in self.all_sem_classes:
            if direction == 'left':
                labels[sem_id] = labels_dict['dis_to_left']
            if direction == 'right':
                labels[sem_id] = labels_dict['dis_to_right']

        for sem_id in self.all_sem_classes:
            labels[sem_id] = torch.tensor([labels[sem_id]], dtype=torch.float32)
        return images, labels