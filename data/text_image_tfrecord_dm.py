# 20220424
# hexiaohao
# 定制化 pytorch_lightning 的 text 和 image 数据集
# 读入 tfrecord 数据

# Originally found in https://github.com/lucidrains/DALLE-pytorch
import os
from pathlib import Path
from random import randint, choice

import cv2
import PIL
import argparse
import clip
import torch

from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule


# 定义 pytorch 类型的 TextImageDataset
class TextImageDataset(Dataset):
    def __init__(self,
                 tfrecord_pattern: str,
                 index_pattern: str,
                 splits: dict,
                 description: dict,
                 decode_tfrecord,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 custom_tokenizer=False
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
        """
        super().__init__()

        text_files = []
        image_files = []

        room_ids = []
        anchor_ids = []
        uids = []
        chnids = []
        req_froms = []

        # 解析 tfrecord 数据
        tfrecord_dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description, transform=decode_tfrecord, infinite=False)
        for tf_data in iter(tfrecord_dataset):
            text_files.append(tf_data['asr'])
            image_files.append(tf_data['img'])
            room_ids.append(tf_data['room_id'])
            anchor_ids.append(tf_data['anchor_id'])
            uids.append(tf_data['uid'])
            chnids.append(tf_data['chnid'])
            req_froms.append(tf_data['req_from'])

        self.shuffle = shuffle

        self.resize_ratio = resize_ratio

        # 定义图像 transform 函数
        self.image_transform = T.Compose([
            T.Lambda(self.fix_img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.custom_tokenizer = custom_tokenizer

    def __len__(self):
        return len(self.image_files)
    
    # 对 pil 图像格式进行处理
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    # 获取数据集 idx 的随机数
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    # 序列循环 sample
    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    # 顺序跳过下标为 ind 的 sample
    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    # 标准的根据 idx 获取 item
    def __getitem__(self, ind):
        # raw text
        text_file = self.text_files[ind]
        # pil image
        image_file = self.image_files[ind]

        # 对 description 进行 tokenized
        tokenized_text = text_file if self.custom_tokenizer else clip.tokenize(text_file)[0]

        # 从 image_file 读取图像
        try:
            image_tensor = self.image_transform(image_file)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return image_tensor, tokenized_text

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 tfrecord_pattern: str,
                 index_pattern: str,
                 splits: dict,
                 description: dict,
                 decode_tfrecord,
                 batch_size: int,
                 num_workers=0,
                 image_size=224,
                 resize_ratio=0.75,
                 shuffle=False,
                 maxlen=32,
                 custom_tokenizer=None
                 ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            batch_size (int): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
        """
        super().__init__()
        self.tfrecord_pattern = tfrecord_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.decode_tfrecord = decode_tfrecord

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.maxlen = maxlen
        self.custom_tokenizer = custom_tokenizer
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # # 数据集 path
        parser.add_argument('--tfrecord_pattern', type=str, required=True, help='directory of your tfrecord_pattern')
        parser.add_argument('--index_pattern', type=str, required=True, help='directory of your index_pattern')

        # bs
        parser.add_argument('--batch_size', type=int, help='size of the batch')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')

        # 图像大小
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--resize_ratio', type=float, default=0.75, help='minimum size of images during random crop')
        parser.add_argument('--shuffle', type=bool, default=False, help='whether to use shuffling during sampling')

        # 文本 tokenizer 限制长度
        parser.add_argument('--maxlen', type=int, default=32, help='max_length of the text tokenizer')

        return parser

    # setup 阶段获取 dataset
    def setup(self, stage=None):
        self.dataset = TextImageDataset(self.tfrecord_pattern, self.index_pattern, self.splits, self.description, self.decode_tfrecord, image_size=self.image_size, resize_ratio=self.resize_ratio, shuffle=self.shuffle, custom_tokenizer=not self.custom_tokenizer is None)
    
    # 根据 dataset 获取 train_dataloader
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True , collate_fn=self.dl_collate_fn)
    
    # 对 batch 数据进行对齐
    def dl_collate_fn(self, batch):
        if self.custom_tokenizer is None:
            return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch])
        else:
            return torch.stack([row[0] for row in batch]), self.custom_tokenizer([row[1] for row in batch], max_length=self.maxlen, padding='max_length', truncation=True, return_tensors="pt")
