import numpy as np
from PIL import Image
import h5py

import scipy.sparse
import scipy.io

def sparse_to_dense(sparse_array):
    dense_array = []
    for i in range(sparse_array.shape[0]):
        dense_matrix = sparse_array[i, 0].toarray()
        dense_array.append(dense_matrix)
    return np.array(dense_array)
    
def load_segmentation_from_mat(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    gt_cls = mat_data['GTinst']
    boundaries = gt_cls['Boundaries'][0, 0] 
    dense_boundaries = sparse_to_dense(boundaries)
    return dense_boundaries
    


def get_sbd_gtfiles(filenames):

   
    label = load_segmentation_from_mat(filenames)
   
    label = label.transpose(1, 2, 0)
    
    return label


