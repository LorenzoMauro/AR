import numpy as np
import pprint
import config
import h5py
import cv2


class prep_dataset:
    def __init__(self):
        self.base_path = "dataset/preprocessed/"

    def get_matrix(self, path, frame):
        video_name = path.split('/')[-1].split('.')[0]
        frame_path = self.base_path + video_name + '/' + str(frame)
        im = cv2.imread(frame_path + '_rgb.jpg', cv2.IMREAD_UNCHANGED)
        flow_1 = cv2.imread(frame_path + '_flow_1.jpg', cv2.IMREAD_UNCHANGED)
        flow_2 = cv2.imread(frame_path + '_flow_2.jpg', cv2.IMREAD_UNCHANGED)
        pafMat = cv2.imread(frame_path + '_pafMat.jpg', cv2.IMREAD_UNCHANGED)
        heatMat =cv2.imread(frame_path + '_heatMat.jpg', cv2.IMREAD_UNCHANGED) 
        frame_matrix = np.zeros(shape=(368, 368, 7), dtype=float)
        frame_matrix[:, :, :3] = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
        frame_matrix[:, :, 5] = cv2.normalize(flow_1, None, 0, 255, cv2.NORM_MINMAX)
        frame_matrix[:, :, 6] = cv2.normalize(flow_2, None, 0, 255, cv2.NORM_MINMAX)
        frame_matrix[:, :, 3] = cv2.normalize(pafMat, None, 0, 255, cv2.NORM_MINMAX)
        frame_matrix[:, :, 4] = cv2.normalize(heatMat, None, 0, 255, cv2.NORM_MINMAX)
        resized = cv2.resize(frame_matrix, dsize=(config.out_H, config.out_W), interpolation=cv2.INTER_CUBIC)
        resized = resized.astype(np.uint8)
        return resized
