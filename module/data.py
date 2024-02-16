from io import BytesIO
import json
import os
import pickle


import numpy as np
import skimage.io
import torch
import torch_geometric
from torch_geometric.data import Data 
import transformers
from ml_collections import ConfigDict
from collections import defaultdict
from PIL import Image
from skimage.color import gray2rgb, rgba2rgb
from torchvision import transforms
from tqdm import trange


class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, filename):
        self.root = os.path.join(root, mode)
        self.json_file = json.load(open(os.path.join(self.root, filename)))
        self.mode = mode
        self.triples = self._prepro()

    def __getitem__(self, idx):
        return self.triples[idx]
    
    def __len__(self):
        return len(self.triples)
    
    def _prepro(self):
        ent2id = json.load(open(os.path.join(self.root, 'entity2ids_zsl.json')))
        rel2id = json.load(open(os.path.join(self.root, 'relation2ids.json')))
        triples = list()
        for rel in self.json_file.keys():
            for triple in self.json_file[rel]:
                h, r, t = triple
                triples.append([ent2id[h], rel2id[r], ent2id[t]])
        return triples

class MMKGDataset(torch_geometric.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = True

        config.image_only = False
        config.text_only = False
        config.struct_only = False
        config.tokenize = True
        config.tokenizer = "   "### load your own downloaded bert tokenizer path here
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
    
    def __init__(self, config, train_file, name, root, mode, mm_info, rel_des_file, transform=None, pre_transform=None):
        self.config = self.get_default_config(config)
        assert self.config.image_only != self.config.text_only or (self.config.image_only == self.config.text_only and self.config.text_only == False)
        self.name = name
        self.root = root
        self.train_file = train_file
        self.rel_description_file = rel_des_file
        self.num_relations = len(rel_des_file)
        
        super().__init__(os.path.join(root, mode), transform, pre_transform)

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
        self.mm_info = mm_info

        self.struc_dataset = self.get(0)
        #self._multimodal_prepro()

    @property
    def raw_file_names(self):
        return [self.train_file, 'entity2ids_zsl.json', 'relation2ids.json']
    
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

    def _image_prepro(self, images_ori):
        # images = []
        # for image_ori in images_ori:
        #     with BytesIO(image_ori) as fin:
        #         image = skimage.io.imread(fin)
        #     if len(image.shape) == 2:
        #         image = gray2rgb(image)
        #     elif image.shape[-1] == 4:
        #         image = rgba2rgb(image)

        #     image = (
        #         self.transform_image(Image.fromarray(np.uint8(image))).permute(1, 2, 0)
        #     )
        #     # image = image.astype(np.float32)
        #     images.append(image)
        # return torch.cat(images, dim=0)
        with BytesIO(images_ori) as fin:
            image = skimage.io.imread(fin)
        if len(image.shape) == 2:
            image = gray2rgb(image)
        elif image.shape[-1] == 4:
            image = rgba2rgb(image)

        image = (
            self.transform_image(Image.fromarray(np.uint8(image))).permute(1, 2, 0)
        )
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
    
    def generate_batch(self, node_list, batch_rels):
        batch_unprocessed_data = [self.mm_info[idx] for idx in node_list]
        batch_rel_des = [self.rel_description_file[idx] for idx in batch_rels]
        mm_batch = defaultdict(list)
        for node_info in batch_unprocessed_data:
            if node_info.__len__() == 2:
                images_ori, text_ori = node_info
                if not self.config.text_only:
                    image = self._image_prepro(images_ori)
                    mm_batch['image'].append(image)
                    if self.config.image_only:
                        continue
            else:
                text_ori = node_info[0]
                if not self.config.text_only:
                    image = torch.empty(self.config.image_size, self.config.image_size, 3)
                    torch.nn.init.xavier_uniform_(image)
                    image *= 10
                    mm_batch['image'].append(image)
                    if self.config.image_only:
                        continue
            
            try:
                text, text_padding_mask = self._text_prepro(text_ori, self.config.tokenizer_max_length)
                mm_batch['text'].append(text)
                mm_batch['text_padding_mask'].append(text_padding_mask)
            except:
                mm_batch['text'].append(text_ori)
        
        for des in batch_rel_des:
            des_tokens, des_padding_mask = self._text_prepro(des, self.config.unpaired_tokenizer_max_length)
            mm_batch['rel_des'].append(des_tokens)
            mm_batch['rel_des_padding_mask'].append(des_padding_mask)

        if len(mm_batch['image']) != 0:
            mm_batch['image'] = torch.stack(mm_batch['image'], dim = 0).to(torch.float32)
        else:
            mm_batch['image'] = torch.tensor(np.array(mm_batch['image']), dtype=torch.int32)
        mm_batch['text'] = torch.tensor(np.array(mm_batch['text']), dtype=torch.int32)
        mm_batch['text_padding_mask'] = torch.tensor(np.array(mm_batch['text_padding_mask']), dtype=torch.float32)
        mm_batch['rel_des'] = torch.tensor(np.array(mm_batch['rel_des']), dtype=torch.int32)
        mm_batch['rel_des_padding_mask'] = torch.tensor(np.array(mm_batch['rel_des_padding_mask']), dtype=torch.float32)
        return mm_batch

        for node_info in batch_unprocessed_data:
            if node_info.__len__() == 1:
                images_ori = node_info[0]
                if not self.config.text_only:
                    image = self._image_prepro(images_ori)
                    mm_batch['image'].append(image)
                    if self.config.image_only:
                        continue
            else:
                if not self.config.text_only:
                    image = torch.empty(self.config.image_size, self.config.image_size, 3)
                    torch.nn.init.xavier_uniform_(image)
                    image *= 10
                    mm_batch['image'].append(image)
                    if self.config.image_only:
                        continue

        if len(mm_batch['image']) != 0:
            mm_batch['image'] = torch.stack(mm_batch['image'], dim = 0).to(torch.float32)
        else:
            mm_batch['image'] = torch.tensor(np.array(mm_batch['image']), dtype=torch.int32)
        mm_batch['text'] = torch.tensor(np.array(mm_batch['text']), dtype=torch.int32)
        mm_batch['text_padding_mask'] = torch.tensor(np.array(mm_batch['text_padding_mask']), dtype=torch.float32)
        return mm_batch
    
class MultiModalKnowledgeGraphDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.start_index = 0
        config.max_length = int(1e9)
        config.random_start = True

        config.image_only = False
        config.text_only = False
        config.struct_only = False
        config.tokenize = True
        config.tokenizer = " " ### load your own downloaded bert tokenizer path here
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
    
    def __init__(self, config, e2id, r2id, triples, mm_info, rel_des_file):
        self.config = self.get_default_config(config)
        self.triples = triples
        if triples.__len__() == 3:
            h, r, t = triples
            self.triples = [[h_i, r_i, t_i] for h_i, r_i, t_i in zip(h, r, t)]
        self.rel_descriptions = rel_des_file
        self.mm_info = mm_info
        self.e2id = e2id
        self.r2id = r2id
        self.num_nodes = len(e2id)
        self.num_relations = len(r2id)

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
        triple = self.triples[idx]
        h, r, t = triple
        description = self.rel_descriptions[r]
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
        
        rel_des, rel_des_pad_mask = self._text_prepro(description, self.config.unpaired_tokenizer_max_length)
        image_head = head_batch['image']
        image_tail = tail_batch['image']
        text_head = head_batch['text']
        text_tail = tail_batch['text']
        text_pad_mask_head = head_batch['text_padding_mask']
        text_pad_mask_tail = tail_batch['text_padding_mask']
        return torch.tensor(triple), image_head, text_head, text_pad_mask_head, image_tail, text_tail, text_pad_mask_tail, rel_des, rel_des_pad_mask
    
    def __len__(self):
        return len(self.triples)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
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
                unpaired_text, unpaired_text_padding_mask = self._text_prepro(text_ori, self.config.tokenizer_max_length)
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
            self.transform_image(Image.fromarray(np.uint8(image))).permute(1, 2, 0)
        )
        #image = image.astype(np.float32)
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
    
    def get_batch(self, triples : list):
        '''
            triples should contain three list: heads relations tails
        '''
        batch = defaultdict(list)
        hs, rs, ts = triples
        for h, r, t in zip(hs, rs, ts):
            batch['triples'].append([h, r, t])

            description = self.rel_descriptions[r]
            if self.mm_info[h].__len__() == 2:
                image_ori, text_ori = self.mm_info[h]
            else:
                image_ori, text_ori = None, self.mm_info[h][0]

            head_batch = self._multimodal_prepro(image_ori, text_ori)
            batch['image'].append(head_batch['image'])
            batch['text'].append(head_batch['text'])
            batch['text_padding_mask'].append(head_batch['text_padding_mask'])

            rel_des, rel_des_pad_mask = self._text_prepro(description, self.config.unpaired_tokenizer_max_length)
            batch['rel_des'].append(rel_des)
            batch['rel_des_padding_mask'].append(rel_des_pad_mask)

        batch['triples'] = torch.tensor(np.array(batch['triples']), dtype=torch.int64)
        if len(batch['image']) != 0:
            batch['image'] = torch.stack(batch['image'], dim = 0).to(torch.float32)
        else:
            batch['image'] = torch.tensor(np.array(batch['image']), dtype=torch.int32)
        batch['text'] = torch.tensor(np.array(batch['text']), dtype=torch.int32)
        batch['text_padding_mask'] = torch.tensor(np.array(batch['text_padding_mask']), dtype=torch.float32)
        batch['rel_des'] = torch.tensor(np.array(batch['rel_des']), dtype=torch.int32)
        batch['rel_des_padding_mask'] = torch.tensor(np.array(batch['rel_des_padding_mask']), dtype=torch.float32)
        return batch
    
