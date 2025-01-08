import numpy as np
from PIL import Image
import h5py


def get_bsds_gtfiles(filenames):
    '''
    edge = Image.open(filenames, 'r')
    label = np.array(edge, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label /= 255.
    label[label >= 0.3] = 1
    #label = torch.from_numpy(label).float()
    return label
    '''
    # RINDNet/dataloaders/datasets/bsds_hd5.py
    h = h5py.File(filenames, 'r')
    edge = np.squeeze(h['label'][...])
    #print("here!")
    #print(edge.shape)
    label = edge.astype(np.float32)
    #label = torch.from_numpy(label).float()
    #print(filenames)
    label = label[1:5,:,:]
    #print(label.shape)
    label = label.transpose(1,2,0)
    #print(label.shape)
    return label

def get_bsds_gtfiles_bythr(filenames,thr):
    '''
    edge = Image.open(filenames, 'r')
    label = np.array(edge, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label /= 255.
    label[label >= thr] = 1
    #label = torch.from_numpy(label).float()
    return label
    '''

    h = h5py.File(filenames, 'r')
    edge = np.squeeze(h['label'][...])
    #print(edge.shape)
    label = edge.astype(np.float32)
    #label = torch.from_numpy(label).float()
    #print(filenames)
    return label

