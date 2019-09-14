"""
Custom version of extract_features that crops cells on the fly.

Author: Matej Usaj
Email: m.usaj@utoronto.ca
Copyright (C) 2019 Matej Usaj
"""
import fnmatch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import tensorflow as tf
import skimage.exposure

from pair_model import Pair_Model
import opts as opt

def read_crops(crop_path, crop_size=64, channels=2, dtype=np.uint16):
    """ Read saved crops

    :param crop_path: path to memmapped data
    :param crop_size: width/height of crops
    :param channels: number of channels
    :param dtype: dtype of saved crops
    :type crop_path: str
    :type crop_size: int
    :type channels: int
    :type dtype: class
    :return: Crops in one data structure
    :rtype: np.ndarray
    """
    fp = np.memmap(crop_path, dtype=dtype, mode='r')
    fp = fp.reshape((-1, channels, crop_size, crop_size))
    return fp


if __name__ == "__main__":
    # Layers to extract single cell features from
    layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # Directory of subfolders of single-cell image crops
    datapath = opt.data_path

    # Location of pretrained weights for the model
    modelpath = opt.checkpoint_path + "model_weights.h5"

    for layer in layers:
        print("Loading the model...")
        # Load pretrained model and set the layer to extract features from
        model = Pair_Model().create_model((opt.im_h, opt.im_w, 2), (opt.im_h, opt.im_w, 1))
        model.load_weights(modelpath)
        intermediate_model = tf.keras.Model(inputs=model.get_layer("x_in").input,
                                            outputs=model.get_layer(layer).output)

        print("Evaluating images...")

        for path, folders, files in os.walk(datapath):
            for file in fnmatch.filter(files, "*.dat"):
                print("Evaluating", file)

                name = file[:-4]  # strip extension

                crops = read_crops(os.path.join(path, file)).astype(np.float32)

                for crop in crops:
                    gfp = skimage.exposure.rescale_intensity(crop[0], out_range=(0, 1))
                    rfp = skimage.exposure.rescale_intensity(crop[1], out_range=(0, 1))

                    # Feed single cell crop into the pretrained model and obtain features
                    x_in = np.stack((gfp, rfp), axis=-1)
                    x_in = np.expand_dims(x_in, axis=0)

                    prediction = intermediate_model.predict([x_in], batch_size=1)

                    prediction = np.squeeze(prediction)
                    prediction = np.max(prediction, axis=(0, 1))

                    # Write features into a file
                    outputfile = opt.checkpoint_path + "yeast_features_" + layer + ".txt"
                    output = open(outputfile, "a")
                    output.write(name)
                    output.write("\t")
                    for feat in prediction:
                        output.write(str(feat))
                        output.write("\t")
                    output.write("\n")
                    output.close()
