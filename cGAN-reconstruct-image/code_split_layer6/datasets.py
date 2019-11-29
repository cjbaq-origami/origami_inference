from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import os
import pickle
import numpy as np

class PairDataset(data.Dataset):
    def __init__(self, data_dir, 
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = data_dir
        
        self.pairs = ['%d.pickle' % i for i in range(50000)]
        #self.pairs = ['%d.pickle' % i for i in range(50)]


    def __getitem__(self, index):
        key = self.pairs[index]
        data = None
        with open(os.path.join(self.data_dir, key), 'rb') as handle:
            data = pickle.load(handle) 
        img, feat = data
        img /= 127.5
        img -= 1.0
        feat = feat.astype(np.float32)
        return img, feat

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    from model import G_NET, D_NET
    import torchvision.transforms as transforms
    netG = G_NET()
    netD = D_NET()
    image_transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = PairDataset('/home/zlin/perturbation/split_19/testData')
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    i = 0
    print(len(dataset))
    for a, b in loader:
        a = a.squeeze(1).permute(0,3,1,2)
        b = b.squeeze(1).permute(0,3,1,2)
        print (a.min(), a.max())
        k = netG(b)
        print (k.min(), k.max())
        kk = netD(a, b)
        kkk = netD.get_cond_logits(kk)
        print(k.shape)
        print(kk.shape)
        print(kkk.shape)
        if i > 10: break
        i += 1
