from io import BytesIO
import json
import os
import pickle

import gcsfs
import h5py
import numpy as np
import skimage.io
import torch
import torchvision
import torch_geometric
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data, Dataset
import transformers
from ml_collections import ConfigDict
from collections import defaultdict
from PIL import Image
from skimage.color import gray2rgb, rgba2rgb
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from tqdm import trange

class ImageTextDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.dataset = ''
        config.path = "./origin_data/FB15K-237/FB15K-237_paired.hdf5"

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_only = False
        config.tokenize = True
        config.tokenizer = "/media/omnisky/sdb/grade2020/cairui/Dawnet/m3ae/pretrain_models/bert-base-uncased-tokenizer"
        config.tokenizer_max_length = 64

        config.transform_type = "pretrain"
        config.image_size = 256

        config.image_normalization = 'imagenet'
        config.custom_image_mean = [0.485, 0.456, 0.406]
        config.custom_image_std = [0.229, 0.224, 0.225]

        config.random_drop_text = 0.0
        config.deterministic_drop_text = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.image_normalization == 'imagenet':
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == 'cc12m':
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == 'none':
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == 'custom':
            self.image_mean = tuple(float(x) for x in self.config.custom_image_mean.split('-'))
            self.image_std = tuple(float(x) for x in self.config.custom_image_std.split('-'))
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError('Unsupported image normalization mode!')

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == "finetune":
            # Use Kaiming's finetune processing
            self.transform = create_transform(
                input_size=self.config.image_size,
                is_training=True,
                color_jitter=True,
                auto_augment=None,
                interpolation="bicubic",
                re_prob=0,
                re_mode=0,
                re_count="const",
                mean=self.image_mean,
                std=self.image_std,
            )
        elif self.config.transform_type == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        elif self.config.transform_type == 'resize_only':
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")

        if self.config.tokenize:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.config.tokenizer
            )

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["jpg"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def drop_text(self, raw_index):
        deterministic_drop = float(raw_index % 100) / 100. < self.config.deterministic_drop_text
        random_drop = np.random.rand() < self.config.random_drop_text
        return deterministic_drop or random_drop

    def __getitem__(self, raw_index):
        index = self.process_index(raw_index)
        with BytesIO(self.h5_file["jpg"][index]) as fin:
            image = skimage.io.imread(fin)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)
        if self.config.image_only:
            return image

        with BytesIO(self.h5_file["caption"][index]) as fin:
            caption = fin.read().decode("utf-8")

        if not self.config.tokenize:
            return image, caption

        if len(caption) == 0 or self.drop_text(raw_index):
            tokenized_caption = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
            return image, tokenized_caption, padding_mask

        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )

        if encoded_caption["input_ids"][0].size == 0:  # Empty token
            tokenized_caption = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_caption = encoded_caption["input_ids"][0]
            padding_mask = 1.0 - encoded_caption["attention_mask"][0].astype(np.float32)

        return image, tokenized_caption, padding_mask

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def text_length(self):
        return self.config.tokenizer_max_length


class ImageNetDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ""
        config.partition = "train"
        config.image_only = False

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.image_normalization = 'imagenet'
        config.transform_type = "pretrain"
        config.image_size = 256

        config.autoaug = "rand-m9-mstd0.5-inc1"

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.image_normalization == 'imagenet':
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == 'cc12m':
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == 'none':
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == 'custom':
            self.image_mean = tuple(float(x) for x in self.config.custom_image_mean.split('-'))
            self.image_std = tuple(float(x) for x in self.config.custom_image_std.split('-'))
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError('Unsupported image normalization mode!')

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.image_mean, std=self.image_std
                    ),
                ]
            )
        elif self.config.transform_type == "finetune":
            # Use Kaiming's finetune processing
            self.transform = create_transform(
                input_size=self.config.image_size,
                is_training=True,
                color_jitter=True,
                auto_augment=self.config.autoaug,
                interpolation="bicubic",
                re_prob=0,
                re_mode=0,
                re_count="const",
                mean=self.image_mean,
                std=self.image_std,
            )
        elif self.config.transform_type == "plain_finetune":
            # Use supervised training processing of ViT from "Better plain ViT baselines for ImageNet-1k" https://arxiv.org/abs/2205.01580
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.image_mean, std=self.image_std
                    ),
                ]
            )
        elif self.config.transform_type == "linear_prob":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.image_mean, std=self.image_std
                    ),
                ]
            )
        elif self.config.transform_type == "test":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.config.image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(self.config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.image_mean, std=self.image_std
                    ),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["{}_jpg".format(self.config.partition)].shape[0]
            - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def __getitem__(self, index):
        index = self.process_index(index)
        with BytesIO(
            self.h5_file["{}_jpg".format(self.config.partition)][index]
        ) as fin:
            image = skimage.io.imread(fin)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)

        if self.config.image_only:
            return image

        label = self.h5_file["{}_labels".format(self.config.partition)][index]

        return image, label

    def num_classes(self):
        return 1000


class TextDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = "./origin_data/FB15K-237/FB15K-237_unpaired_text.hdf5"

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = True

        config.tokenize = True
        config.tokenizer = "/media/omnisky/sdb/grade2020/cairui/Dawnet/m3ae/pretrain_models/bert-base-uncased-tokenizer"
        config.tokenizer_max_length = 256

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, start_offset_ratio=None):
        self.config = self.get_default_config(config)
        assert self.config.path != ""

        if self.config.path.startswith("gs://"):
            # Loading from GCS
            self.h5_file = h5py.File(
                gcsfs.GCSFileSystem().open(self.config.path, cache_type="block"), "r"
            )
        else:
            self.h5_file = h5py.File(self.config.path, "r")

        if self.config.tokenize:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.config.tokenizer
            )

        if self.config.random_start:
            # Bypass numpy random seed
            self.random_start_offset = np.random.default_rng().choice(len(self))
        elif start_offset_ratio is not None:
            self.random_start_offset = int(len(self) * start_offset_ratio) % len(self)
        else:
            self.random_start_offset = 0

    def __getstate__(self):
        return self.config, self.random_start_offset

    def __setstate__(self, state):
        config, random_start_offset = state
        self.__init__(config)
        self.random_start_offset = random_start_offset

    def __len__(self):
        return min(
            self.h5_file["text"].shape[0] - self.config.start_index,
            self.config.max_length,
        )

    def process_index(self, index):
        index = (index + self.random_start_offset) % len(self)
        return index + self.config.start_index

    def __getitem__(self, raw_index):
        index = self.process_index(raw_index)

        with BytesIO(self.h5_file["text"][index]) as fin:
            text = fin.read().decode("utf-8")

        if not self.config.tokenize:
            return text

        if len(text) == 0:
            tokenized = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
            return tokenized, padding_mask

        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )

        if encoded_text["input_ids"][0].size == 0:  # Empty token
            tokenized_text = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_text = encoded_text["input_ids"][0]
            padding_mask = 1.0 - encoded_text["attention_mask"][0].astype(np.float32)

        return tokenized_text, padding_mask

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


class MMKGDataset(torch_geometric.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = True

        config.image_only = False
        config.tokenize = True
        config.tokenizer = "/media/omnisky/sdb/grade2020/cairui/Dawnet/m3ae/pretrain_models/bert-base-uncased-tokenizer"
        config.tokenizer_max_length = 64
        config.unpaired_tokenizer_max_length = 320
        
        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.transform_type = "pretrain"
        config.image_size = 256

        config.image_normalization = 'imagenet'
        config.custom_image_mean = [0.485, 0.456, 0.406]
        config.custom_image_std = [0.229, 0.224, 0.225]

        config.random_drop_text = 0.0
        config.deterministic_drop_text = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config
    
    def __init__(self, config, train_file, name, root, transform=None, pre_transform=None):
        self.config = self.get_default_config(config)
        self.name = name
        self.root = root
        self.train_file = train_file
        
        super().__init__(root, transform, pre_transform)

        if self.config.image_normalization == 'imagenet':
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == 'cc12m':
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == 'none':
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == 'custom':
            self.image_mean = tuple(float(x) for x in self.config.custom_image_mean.split('-'))
            self.image_std = tuple(float(x) for x in self.config.custom_image_std.split('-'))
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError('Unsupported image normalization mode!')

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform_image = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.config.tokenizer
        )
        with open(os.path.join(root, 'MultiModalInfo.pkl'), 'rb') as fin:
            self.mm_info = pickle.load(fin)

        self.struc_dataset = self.get(0)
        self.preprocessed = False
        #self._multimodal_prepro()

    @property
    def raw_file_names(self):
        return [self.train_file, 'entity2ids.json', 'relation2ids_allrel.json']
    
    @property
    def processed_file_names(self):
        return [f'{self.name}.pt']

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    @property
    def num_nodes(self):
        return self.get(0).num_nodes
    
    def download(self):
        pass

    def process(self):
        train_tasks = json.load(open(os.path.join(self.root, self.raw_file_names[0])))
        ent2id = json.load(open(os.path.join(self.root, self.raw_file_names[1])))
        rel2id = json.load(open(os.path.join(self.root, self.raw_file_names[2])))
        triples = list()
        for rel in train_tasks.keys():
            for triple in train_tasks[rel]:
                h, r, t = triple
                triples.append([ent2id[h], rel2id[r], ent2id[t]])
        
        # num_nodes = max(ent2id.values())
        # num_relations = max(rel2id.values())
        edge_index = torch.tensor([[h, t] for h, _, t in triples], dtype= torch.long).t().contiguous()
        edge_type = torch.tensor([r for _, r, _ in triples], dtype=torch.long)

        data = Data(edge_index=edge_index, edge_type=edge_type)
        #processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        torch.save((data, None), self.processed_paths[0])
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))[0]
        return data

    def get_struc_dataset(self):
        return self.struc_dataset
    
    def get_mm_info(self):
        return self.mm_info
    
    def multimodal_prepro(self):
        print('Making multimodal infomation preprocessing!')

        def drop_text(raw_index):
            deterministic_drop = float(raw_index % 100) / 100. < self.config.deterministic_drop_text
            random_drop = np.random.rand() < self.config.random_drop_text
            return deterministic_drop or random_drop
        
        steps = trange(0, len(self.mm_info))
        for idx, info in zip(steps, self.mm_info):
            image_ori, text = None, None
            try:
                image_ori, text = info
            except:
                text = info[0]

            if image_ori is not None:
                image = self._image_prepro(image_ori)
                if self.config.image_only:
                    self.mm_info[idx] = [image]
                    continue
                if not self.config.tokenize:
                    self.mm_info[idx] = [image, text]
                    continue
                text, text_padding_mask = self._text_prepro(text, self.config.tokenizer_max_length)
                self.mm_info[idx] = [image, text, text_padding_mask]
            else:
                if not self.config.tokenize:
                    self.mm_info[idx] = [text]
                    continue
                text, text_padding_mask = self._text_prepro(text, self.config.unpaired_tokenizer_max_length)
                self.mm_info[idx] = [text, text_padding_mask]
            # if len(text) == 0 or drop_text(idx):
            #     tokenized_caption = np.zeros(self.config.tokenizer_max_length, dtype=np.int32)
            #     padding_mask = np.ones(self.config.tokenizer_max_length, dtype=np.float32)
            #     self.mm_info[idx] = [image, tokenized_caption, padding_mask]
        self.preprocessed = True

        with open(os.path.join(self.root, 'multimodal_processed.pkl'), 'wb') as fout:
            pickle.dump(self.mm_info, fout)
        print('Finish multimodal infomation preprocessing!')

    def _image_prepro(self, image_ori):
        with BytesIO(image_ori) as fin:
            image = skimage.io.imread(fin)
        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform_image(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)

        return image
    
    def _text_prepro(self, text, tokenizer_max_length):
        if not self.config.tokenize:
            return text
        
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )       
        if encoded_text["input_ids"][0].size == 0:  # Empty token
            tokenized_text = np.zeros(tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_text = encoded_text["input_ids"][0]
            padding_mask = 1.0 - encoded_text["attention_mask"][0].astype(np.float32)
        return tokenized_text, padding_mask
    
    def generate_batch(self, node_list):
        batch_unprocessed_data = [self.mm_info[idx] for idx in node_list]
        mm_batch = defaultdict(list)
        #images, tokenized_texts, padding_masks, unpaired_texts, unpaired_text_padding_masks = list(), list(), list(), list(), list()
        features = list()
        for node_info in batch_unprocessed_data:
            #assert len(node_info) == 2 or len(node_info) == 1
            if self.preprocessed == False:
                if node_info.__len__() == 2:
                    image, text = node_info
                    image = self._image_prepro(image)
                    mm_batch['image'].append(image) 
                    if self.config.image_only:
                        features.append([image])
                        continue
                    try:
                        text, text_padding_mask = self._text_prepro(text, self.config.tokenizer_max_length)
                        mm_batch['text'].append(text)
                        mm_batch['text_padding_mask'].append(text_padding_mask)
                        features.append([image, text, text_padding_mask])
                    except:
                        text = self._text_prepro(text, self.config.tokenizer_max_length)
                        features.append([image, text])
                else:
                    text = node_info[0]
                    try:
                        text, text_padding_mask = self._text_prepro(text, self.config.unpaired_tokenizer_max_length)
                        mm_batch['unpaired_text'].append(text)
                        mm_batch['unpaired_text_padding_mask'].append(text_padding_mask)
                        features.append([text, text_padding_mask])
                    except:
                        text = self._text_prepro(text, self.config.unpaired_tokenizer_max_length)
                        features.append([text])
            else: 
                if node_info.__len__() == 2:
                    text, text_padding_mask = node_info
                    mm_batch['unpaired_text'].append(text)
                    mm_batch['unpaired_text_padding_mask'].append(text_padding_mask)
                elif node_info.__len__() == 3:
                    image, text, text_padding_mask = node_info
                    mm_batch['image'].append(image)
                    mm_batch['text'].append(text)
                    mm_batch['text_padding_mask'].append(text_padding_mask) 

        mm_batch['image'] = torch.tensor(np.array(mm_batch['image']), dtype=torch.float32)
        mm_batch['text'] = torch.tensor(np.array(mm_batch['text']), dtype=torch.int32)
        mm_batch['text_padding_mask'] = torch.tensor(np.array(mm_batch['text_padding_mask']), dtype=torch.float32)
        mm_batch['unpaired_text'] = torch.tensor(np.array(mm_batch['unpaired_text']), dtype=torch.int32)
        mm_batch['unpaired_text_padding_mask'] = torch.tensor(np.array(mm_batch['unpaired_text_padding_mask']), dtype=torch.float32)
        return mm_batch, features
    

class MultiModalKnowledgeGraphDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = True

        config.image_only = False
        config.tokenize = True
        config.tokenizer = "/media/omnisky/sdb/grade2020/cairui/Dawnet/m3ae/pretrain_models/bert-base-uncased-tokenizer"
        config.tokenizer_max_length = 64
        config.unpaired_tokenizer_max_length = 320
        
        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = False

        config.transform_type = "pretrain"
        config.image_size = 256

        config.image_normalization = 'imagenet'
        config.custom_image_mean = [0.485, 0.456, 0.406]
        config.custom_image_std = [0.229, 0.224, 0.225]

        config.random_drop_text = 0.0
        config.deterministic_drop_text = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config
    
    def __init__(self, config, root, graph_dataset):
        self.config = self.get_default_config(config)
        self.root = root
        with open(os.path.join(self.root, 'MultiModalInfo.pkl'), 'rb') as fin:
            self.mm_info = pickle.load(fin)
        self.num_nodes = graph_dataset.num_nodes
        self.graph_dataset = graph_dataset
        #self.triples, self.num_nodes = self._prepro(train_tasks)

        if self.config.image_normalization == 'imagenet':
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
        elif self.config.image_normalization == 'cc12m':
            self.image_mean = (0.5762, 0.5503, 0.5213)
            self.image_std = (0.3207, 0.3169, 0.3307)
        elif self.config.image_normalization == 'none':
            self.image_mean = (0.0, 0.0, 0.0)
            self.image_std = (1.0, 1.0, 1.0)
        elif self.config.image_normalization == 'custom':
            self.image_mean = tuple(float(x) for x in self.config.custom_image_mean.split('-'))
            self.image_std = tuple(float(x) for x in self.config.custom_image_std.split('-'))
            assert len(self.image_mean) == len(self.image_std) == 3
        else:
            raise ValueError('Unsupported image normalization mode!')

        if self.config.transform_type == "pretrain":
            # Use Kaiming's simple pretrain processing
            self.transform_image = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.config.image_size,
                        scale=(0.2, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )
        else:
            raise ValueError("Unsupported transform_type!")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.config.tokenizer
        )

    def __getitem__(self, idx):
        # select_node = idx % self.num_nodes
        # sub_graph = NeighborLoader(self.graph_dataset, num_neighbors=[3, 3], input_nodes=torch.Tensor([select_node]).to(torch.long))
        triple = self.triples[idx]
        h, r, t = triple
        if self.mm_info[h].__len__() == 2:
            image_ori, text_ori = self.mm_info[h]
        else:
            image_ori, text_ori = None, self.mm_info[h][0]
        head_batch = self._multimodal_prepro(image_ori, text_ori)

        if self.mm_info[t].__len__() == 2:
            image_ori, text_ori = self.mm_info[t]
        else:
            image_ori, text_ori = None, self.mm_info[t][0]
        tail_batch = self._multimodal_prepro(image_ori, text_ori)
        
        images = [head_batch['image'], tail_batch['image']]
        texts = [head_batch['text'], tail_batch['text']]
        text_padding_masks = [head_batch['text_padding_mask'], tail_batch['text_padding_mask']]
        return triple, images, texts, text_padding_masks
    
    def __len__(self):
        return len(self.triples)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    def _prepro(self, train_tasks_name):
        ent2id = json.load(open(os.path.join(self.root, 'entity2ids.json')))
        num_nodes = len(ent2id)
        rel2id = json.load(open(os.path.join(self.root, 'relation2ids_allrel.json')))
        train_tasks = json.load(open(os.path.join(self.root, train_tasks_name)))
        triples = list()
        for rel in train_tasks.keys():
            for triple in train_tasks[rel]:
                h, r, t = triple
                triples.append([ent2id[h], rel2id[r], ent2id[t]])
        return triples, num_nodes
    
    def _multimodal_prepro(self, image_ori, text_ori):
        batch = dict()
        if image_ori is not None:
            batch['ispaired'] = True
            image = self._image_prepro(image_ori)
            batch['image'] = image
            if self.config.image_only:
                return batch
            try:
                text, text_padding_mask = self._text_prepro(text_ori, self.config.tokenizer_max_length)
                batch['text'] = text
                batch['text_padding_mask'] = text_padding_mask
            except:
                batch['text'] = text_ori
        else:
            batch['image'] = np.random.randn(256, 256, 3)
            batch['ispaired'] = False
            try:
                unpaired_text, unpaired_text_padding_mask = self._text_prepro(text_ori, self.config.tokenizer_max_length )
                batch['text'] = unpaired_text
                batch['text_padding_mask'] = unpaired_text_padding_mask
            except:
                batch['text'] = text_ori
        return batch
            
    def _image_prepro(self, image_ori):
        with BytesIO(image_ori) as fin:
            image = skimage.io.imread(fin)
        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform_image(Image.fromarray(np.uint8(image))).permute(1, 2, 0).numpy()
        )
        image = image.astype(np.float32)
        return image
    
    def _text_prepro(self, text, tokenizer_max_length):
        if not self.config.tokenize:
            return text
        
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_max_length,
            return_tensors="np",
            add_special_tokens=False,
        )       
        if encoded_text["input_ids"][0].size == 0:  # Empty token
            tokenized_text = np.zeros(tokenizer_max_length, dtype=np.int32)
            padding_mask = np.ones(tokenizer_max_length, dtype=np.float32)
        else:
            tokenized_text = encoded_text["input_ids"][0]
            padding_mask = 1.0 - encoded_text["attention_mask"][0].astype(np.float32)
        return tokenized_text, padding_mask


if __name__ == '__main__':
    # device = 'cpu'
    # graph_dataset = MMKGDataset(config=MMKGDataset.get_default_config(), name='FB15K-237', root='../origin_data/FB15K-237', device=device)
    # graph_dataset.generate_batch(torch.arange(1, 100))
    dataset = MultiModalKnowledgeGraphDataset(MultiModalKnowledgeGraphDataset.get_default_config(), '../origin_data/FB15K-237', 'train_tasks_all.json')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        drop_last=True,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
    )
    