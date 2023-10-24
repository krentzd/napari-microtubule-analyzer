import urllib.request
import zipfile
import io
import os
import tifffile
import numpy as np

def zip_reader_from_url(url):
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, 'r')

    file_names = zip_file_object.namelist()
    file_list = [zip_file_object.open(file) for file in file_names[1:]]

    array = [tifffile.imread(io.BytesIO(file.read())) for file in file_list]

    return array, file_names

def read_data(dataset):
    url = f'https://imjoy-s3.pasteur.fr/public/siRNA_sample_data/{dataset}.zip'
    array, file_names = zip_reader_from_url(url)
    data = np.squeeze(np.stack(array))

    metadata_dict = dict()
    metadata_dict['file_paths'] = [os.path.basename(p) for p in file_names[1:]]
    add_kwargs = {"name": file_names[0][:-1], "metadata": metadata_dict}
    layer_type = "image"

    return [(data, add_kwargs, layer_type)]

def make_sample_data():
    return read_data('siAPC') + read_data('siCtrl')
