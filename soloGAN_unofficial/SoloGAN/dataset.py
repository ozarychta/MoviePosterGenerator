
import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
from ezc3d import c3d
import numpy as np
import torch
from utils import generateTargetLabel


class dataset_multi(data.Dataset):
    def __init__(self, opts,phase ='train'):
        self.dataroot = opts.dataroot
        self.num_domains = opts.num_domains

        domains = [chr(i) for i in range(ord('A'),ord('Z')+1)] # domains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.images = [None]*self.num_domains
        stats = ''
        for i in range(self.num_domains):
            img_dir = os.path.join(self.dataroot, phase + domains[i])
            ilist = os.listdir(img_dir)
            self.images[i] = [os.path.join(img_dir, x) for x in ilist]
            stats += '{}: {}'.format(domains[i], len(self.images[i]))
        stats += ' images'
        self.dataset_size = max([len(self.images[i]) for i in range(self.num_domains)])

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if opts.phase == 'train':
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))

        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)

        return

    def __getitem__(self, index):
        simgs ,timgs =[],[]
        sourceID, targetID = [],[]
        s = np.random.randint(0, self.num_domains)
        img = self.load_img((self.images[s][random.randint(0, len(self.images[s]) - 1)]))
        sourceID.append(s)
        simgs.append(img)
        t = generateTargetLabel(s, self.num_domains)
        img = self.load_img((self.images[t][random.randint(0, len(self.images[t]) - 1)]))
        targetID.append(t)
        timgs.append(img)

        return simgs,timgs,sourceID,targetID

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        return img

    def __len__(self):
        return self.dataset_size

class dataset_unpair(data.Dataset):
    def __init__(self, conf):
        self.dataroot = conf['data_root']
        # A
        images_A = os.listdir(os.path.join(self.dataroot + 'trainA'))
        self.A = [os.path.join(self.dataroot + 'trainA', x) for x in images_A if x.endswith('.c3d')]
        # B
        images_B = os.listdir(os.path.join(self.dataroot + 'trainB'))
        self.B = [os.path.join(self.dataroot + 'trainB', x) for x in images_B if x.endswith('.c3d')]

        # print(self.A)

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = conf['input_dim_a']
        self.input_dim_B = conf['input_dim_b']
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_img(self.A[index], self.input_dim_A)
            data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
        else:
            data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
            data_B = self.load_img(self.B[index], self.input_dim_B)
        return data_A, data_B

    def load_img(self, img_name, input_dim):
        sequence = c3d(img_name)
        point_data = sequence['data']['points'][0:3,:,:]/1500
        img = torch.from_numpy(point_data).float()
        return img



    def __len__(self):
        return self.dataset_size





# import os
# import torch.utils.data as data
# from PIL import Image
# from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
# import random
# from ezc3d import c3d
# import numpy as np
# import torch
#
#
# class dataset_unpair(data.dataset):
#     def __init__(self, conf):
#         self.dataroot = conf['data_root'] #opts.dataroot
#         # A
#         images_A = os.listdir(os.path.join(self.dataroot + 'trainA'))
#         self.A = [os.path.join(self.dataroot, + 'trainA', x) for x in images_A if x.endswith('.c3d')]
#
#         # B
#         images_B = os.listdir(os.path.join(self.dataroot + 'trainB'))
#         self.B = [os.path.join(self.dataroot + 'trainB', x) for x in images_B if x.endswith('.c3d')]
#
#         self.A_size = len(self.A)
#         self.B_size = len(self.B)
#         self.dataset_size = max(self.A_size, self.B_size)
#         self.input_dim_A = conf['new_size_a']
#         self.input_dim_B = conf['new_size_b']
#         return
#
#     def __getitem__(self, index):
#         if self.dataset_size == self.A_size:
#             data_A = self.load_img(self.A[index], self.input_dim_A)
#             data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
#         else:
#             data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
#             data_B = self.load_img(self.B[index], self.input_dim_B)
#         return data_A, data_B
#
#     def load_img(self, img_name, input_dim):
#         sequence = c3d(img_name)
#         point_data = sequence['data']['points'][0:3,:,:] - sequence['data']['points'][0:3,5,:]
#         img = torch.from_numpy(point_data).float()
#         return img
#
#     def __len__(self):
#         return self.dataset_size
