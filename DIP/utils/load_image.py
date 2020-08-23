import os
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import pickle
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader

def load_image_coco(num_image=64, crop_size=256):

    images = np.empty((0, crop_size, crop_size, 3), dtype=np.float32)

    coco_anno = json.load(open('/home/chengao/Dataset/annotations/instances_val2017.json'))
    # img_pil = img_pil[[0, 4, 9, 14, 20, 30, 45, 50, 51, 52, 63, 65, 75, 83, 90, 91],:,:,:]
    for idx in range(len(coco_anno['annotations'])):
        GT       = coco_anno['annotations'][idx]
        image_id = GT['image_id']
        H_box    = [int(i) for i in GT['bbox']]
        category = GT['category_id']
        
        if H_box[2] <= crop_size and H_box[2] > 150 / 256 * crop_size and H_box[3] <= crop_size and H_box[3] > 150 / 256 * crop_size:
            
            im_file = '/home/chengao/Dataset/val2017/' + (str(image_id)).zfill(12) + '.jpg'
            im_data = plt.imread(im_file)
            im_height, im_width, nbands = im_data.shape
            
            height_pad = crop_size - H_box[3]
            width_pad  = crop_size - H_box[2]
            
            x0 = H_box[0] - width_pad  // 2
            x1 = H_box[0] - width_pad  // 2 + crop_size
            y0 = H_box[1] - height_pad // 2
            y1 = H_box[1] - height_pad // 2 + crop_size

            if x0 < 0:
                x1 = x1 - x0
                x0 = 0
            if x1 >= im_width:
                continue

            if y0 < 0:
                y1 = y1 - y0
                y0 = 0
            if y1 >= im_height:
                continue   

            im_data = im_data[y0 : y1, x0 : x1, :].reshape(1, crop_size, crop_size, 3)

            images = np.concatenate((images, im_data), axis=0)

            if len(images) >= num_image:
                return images


    return images

class DIV2KDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        
        self.image_list = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.image_list[idx])
        image = io.imread(img_name)
        image = image / 255.

        if self.transform:
            image = self.transform(image)

        return image
    
class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top  = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)