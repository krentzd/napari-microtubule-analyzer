from .utils import abspath
from ._reader import reader_function

def make_sample_data():

    siAPC_path = abspath(__file__, '../../siRNA_sample_data/siAPC')
    siCtrl_path = abspath(__file__, '../../siRNA_sample_data/siCtrl')

    return reader_function(siAPC_path) + reader_function(siCtrl_path)
