"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.root_dir = path_to_folder
        pairs = []
        for dirs in os.listdir(path_to_folder):
            for file in os.listdir(path_to_folder + '/' + dirs):
                pairs += [(dirs, path_to_folder + '/' + dirs + '/' + file)]

        self.pairs = pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index: int) -> torch.Tensor:      
        label, im_path = self.pairs[index]
        image = Image.open(im_path)

        return self.transform(image)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw', type=str)
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        
        dataiter = iter(dataloader)
        images = dataiter.next()

        fix, axs = plt.subplots(ncols=int(np.sqrt(args.batch_size)),nrows=int(np.sqrt(args.batch_size)), squeeze=False)

        for i, img in enumerate(images):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[np.floor(i/int(np.sqrt(args.batch_size))).astype(int), i % int(np.sqrt(args.batch_size))].imshow(np.asarray(img))
            axs[np.floor(i/int(np.sqrt(args.batch_size))).astype(int), i % int(np.sqrt(args.batch_size))].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()

    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for i in range(2):
            print(f"iter {i}")
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print('Timing:', np.mean(res),'+-',np.std(res))
