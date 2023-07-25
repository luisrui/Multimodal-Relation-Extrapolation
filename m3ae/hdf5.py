import os
import tarfile
import json
import random
import numpy as np
from tqdm import tqdm, trange
import h5py
from PIL import Image

dataset = "FB15K-237"

### The file is for transfering the origin data format into HDF5 format
base_dir = '../origin_data/' + dataset + "/"

def create_paired_hdf5(dataset, entity_image, entity_text):  
    ## For creating datasets of image and text pair from the origin-dataset
    h5_bytes = h5py.special_dtype(vlen=np.dtype('uint8'))
    h5_str = h5py.special_dtype(vlen=str)

    min_length = min(entity_image.__len__(), entity_text.__len__())
    h5_file = h5py.File(f'{base_dir}/{dataset}_paired.hdf5','w')
    h5_file.create_dataset('jpg', (min_length, ), dtype=h5_bytes, chunks=True)
    h5_file.create_dataset('caption', (min_length, ), dtype=h5_str, chunks=True)

    for i, ent in enumerate(tqdm(entity_image.keys())):
        h5_file['jpg'][i] = np.frombuffer(entity_image[ent], dtype='uint8')
        h5_file['caption'][i] = entity_text[ent]
        
    h5_file.close()
    print("create_paired_hdf5 succeed!")

def create_unpaired_text_hdf5(dataset, entity_image, entity_text):
    ## For creating datasets of only text info from the origin-dataset
    h5_bytes = h5py.special_dtype(vlen=np.dtype('uint8'))
    h5_str = h5py.special_dtype(vlen=str)

    length = max(entity_image.__len__(), entity_text.__len__()) - min(entity_image.__len__(), entity_text.__len__())
    h5_file = h5py.File(f'{base_dir}/{dataset}_unpaired_text.hdf5','w')
    h5_file.create_dataset('text', (length, ), dtype=h5_str, chunks=True)

    i = 0
    for ent in tqdm(entity_text.keys()):
        if ent not in entity_image.keys():
            h5_file['text'][i] = entity_text[ent]
            i += 1
        
    h5_file.close()
    print("create_unpaired_text_hdf5 succeed!")

with open(os.path.join(base_dir, "entity2textlong.txt"), 'r') as fin:
    entity_text = dict()
    for line in fin.readlines():
        if line[-1] == '\n':
            ent, text = line[:-1].split('\t')
        else:
            ent, text = line.split('\t')
        entity_text.update({ent : text})

entity_image = dict()
for filename in tqdm(os.listdir(os.path.join(base_dir, "images"))):
    #entity = filename[1:]
    entity = '/' + filename.replace('.', '/')
    assert entity in entity_text.keys()
    images = list()
    for pic in os.listdir(os.path.join(base_dir, "images", filename)):
        try:
            image_path = os.path.join(base_dir, "images", filename, pic)
            #im = np.array(Image.open(image_path).convert('RGB'))
            im = open(image_path, 'rb').read()
            images.append(im)
        except:
            print(pic+" did not load")
    if images.__len__() != 0:
        entity_image.update({entity : random.sample(images, 1)[0]})

create_paired_hdf5(dataset, entity_image, entity_text)
create_unpaired_text_hdf5(dataset, entity_image, entity_text)