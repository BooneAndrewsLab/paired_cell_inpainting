import fnmatch
import os

import numpy as np
import pandas as p
from skimage.io import imread


def crop_image(im, cropped_path, coordinates, crop_size=64):
    """ Crop list of cells from image, save it to disk and return the data.
    Remember to delete returned reference for proper garbage collection just in case.

    :param im: array representation of an image
    :param cropped_path: where to save the cropped data
    :param coordinates: list of (x,y) coordinates (centre points) to crop
    :param crop_size: width and lenght of each crop
    :type im: np.ndarray
    :type cropped_path: str
    :type coordinates: list[tuple[int,int]]
    :type crop_size: int
    :return: cropped cells of shape (num_cells, channels, radius*2, radius*2)
    :rtype: np.ndarray
    """
    radius = crop_size // 2

    channels = im.shape[0] if len(im.shape) > 2 else 1  # Get number of channels in this image

    fp = np.memmap(cropped_path, dtype=im.dtype, mode='w+', shape=(len(coordinates), channels, crop_size, crop_size))
    for idx, coord in enumerate(coordinates):
        y, x = coord

        # Handle also images with only one channel so that the cropped files are always 4D
        if channels == 1:
            fp[idx, 0, :, :] = im[x - radius:x + radius, y - radius:y + radius]
        else:
            fp[idx, :, :, :] = im[:, x - radius:x + radius, y - radius:y + radius]
    fp.flush()  # Write data to disk

    return fp


def read_crops(path, crop_size=64, channels=2, dtype=np.uint16):
    """ Read saved crops

    :param path: path to memmapped data
    :param crop_size: width/height of crops
    :param channels: number of channels
    :param dtype: dtype of saved crops
    :type path: str
    :type crop_size: int
    :type channels: int
    :type dtype: class
    :return: Crops in one data structure
    :rtype: np.ndarray
    """
    fp = np.memmap(path, dtype=dtype, mode='r')
    fp = fp.reshape((-1, channels, crop_size, crop_size))
    return fp


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("coordinates_file", help="File containing (x,y) of cells in the images")
    parser.add_argument("in_folder", help="Traverse this top-level folder, crop all images in it")
    parser.add_argument("out_folder", help="Save crops to this folder")
    args = parser.parse_args()

    coordinates = p.read_csv(args.coordinates_file, header=None, names=['image', 'x', 'y'])
    grp = coordinates.groupby('image')
    groups = grp.groups

    top_level_folder = args.in_folder.rstrip('/')
    common_path = os.path.dirname(top_level_folder)

    for path, folders, files in os.walk(top_level_folder):
        for file in fnmatch.filter(files, "*.flex"):
            full_path = os.path.join(path, file)
            relative_path = full_path.replace(common_path, '').lstrip('/')

            if relative_path in groups:
                print(relative_path)
                coords = grp.get_group(relative_path)
                out_path = os.path.join(args.out_folder, relative_path).replace('.flex', '.dat')

                parent_folder = os.path.dirname(out_path)
                if not os.path.exists(parent_folder):
                    os.makedirs(parent_folder)

                img = imread(full_path, plugin='tifffile')
                cropped = crop_image(img, out_path,
                                     [(x, y) for x, y in coords.loc[:, ['x', 'y']].itertuples(index=False)])
                del cropped
