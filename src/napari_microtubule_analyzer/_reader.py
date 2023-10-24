"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import cv2 as cv
import os
import glob
import tifffile

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list) and path.endswith(('.tif', '.tiff', '.png')):
        path = path[0]

    elif os.path.isdir(path):
        pass

    else:
        return None

    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """

    if os.path.isdir(path):
        _paths = sorted(glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.tif')) + glob.glob(os.path.join(path, '*.tiff')))
        # try:
        # array = [cv.imread(_path, cv.IMREAD_GRAYSCALE) for _path in sorted(_paths)]
        # except ValueError:
        array = [tifffile.imread(_path) for _path in sorted(_paths)]

        data = np.squeeze(np.stack(array))

        metadata_dict = dict()
        metadata_dict['file_paths'] = [os.path.basename(p) for p in _paths]

        add_kwargs = {"name": os.path.basename(path), "metadata": metadata_dict}
        # add_kwargs = {"name": os.path.basename(path)}

        layer_type = "image"

    else:
        print(path)
        paths = [path] if isinstance(path, str) else path
        # arrays = [cv.imread(_path, cv.IMREAD_GRAYSCALE) for _path in paths]
        arrays = [tifffile.imread(_path) for _path in paths]
        data = np.squeeze(np.stack(arrays))

        metadata_dict = dict()
        metadata_dict['file_paths'] = [os.path.basename(p) for p in paths]

        add_kwargs = {"name": os.path.basename(path), "metadata": metadata_dict}
        layer_type = "image"

    return [(data, add_kwargs, layer_type)]
