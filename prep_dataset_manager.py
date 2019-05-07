import numpy as np
import pprint
import config
import h5py


class prep_dataset:
    def __init__(self):
        self.data = h5py.File("dataset/preprocessed/dataset.h5", 'r')

    def get_matrix(self, path,frame):
        vide_h5 = self.data.get(path)
        vide_array = np.array(vide_h5)
        frame = vide_array[frame, :, : , :]
        reduced_frame = np.squeeze(frame)
        return reduced_frame
