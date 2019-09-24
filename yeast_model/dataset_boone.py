'''
Scripts to load and prepare yeast single cells for input into the CNN during training.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

import os
from pathlib import Path
from random import randrange

import numpy as np
from PIL import Image

CROP_ST_SIZE = 16384  # 64 * 64 * 2(channels) * 2(uint16)


def get_random_cell(num, not_this):
    while True:
        x = randrange(num)
        if x != not_this:
            return x


class Dataset(object):
    '''We store information about the dataset as a class.'''

    def __init__(self):
        self.image_ids = []
        self.image_info = []

    '''Add an individual image to the class. Called iteratively by add_dataset().
    For memory purposes, this class loads only the paths of each image, and returns the 
    actual images as needed.'''

    def add_image(self, image_id, path, cell_index):
        image_info = {
            "id": image_id,  # Unique integer representation of the image
            "path": path,  # Path where the image is stored
            "index": cell_index
        }
        self.image_info.append(image_info)

    def add_dataset(self, root_dir):
        i = 0  # Used to assign a unique integer index to each image

        root_dir = Path(root_dir)

        for d in root_dir.rglob('*dat'):
            num_cells = d.stat().st_size // CROP_ST_SIZE

            if num_cells == 1:
                continue

            for cell in range(num_cells):
                self.add_image(
                    image_id=i,
                    path=d,
                    cell_index=cell
                )

                i += 1

    '''Load and return the image indexed by the integer given by image_id; also returns the
        name of the image, just for debugging purposes'''

    def load_image_with_label(self, image_id):
        image = self.image_info[image_id]

        crops = np.memmap(image['path'], dtype=np.uint16, mode='r')
        crops = np.reshape(crops, (-1, 2, 64, 64))

        crop = crops[image['index']]
        protein = crop[0]
        brightfield = crop[1]

        label = image['path'].name

        return protein, brightfield, label

    '''Sample a pair for the given image by drawing with equal probability from the folder'''

    def sample_pair_equally(self, image_id):
        image = self.image_info[image_id]

        crops = np.memmap(image['path'], dtype=np.uint16, mode='r')
        crops = np.reshape(crops, (-1, 2, 64, 64))

        num_cells = crops.shape[0]

        if num_cells == 1:
            raise ValueError("%s has only one cell." % (image['path'], ))

        random_cell = crops[get_random_cell(num_cells, image['index'])]
        protein = random_cell[0]
        brightfield = random_cell[1]

        return protein, brightfield

    '''Prepares the dataset file for use.'''

    def prepare(self):
        # Build (or rebuild) everything else from the info dicts.
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)
