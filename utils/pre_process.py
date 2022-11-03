# designed for demo

import numpy as np
import cv2


class PreProcessor(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.input_size = [img_size, img_size]
        

    def __call__(self, image, swap=(2, 0, 1)):
        """
        Input:
            image: (ndarray) [H, W, 3] or [H, W]
            formar: color format
        """
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), np.float32) * 114.
        else:
            padded_img = np.ones(self.input_size, np.float32) * 114.
        # resize
        orig_h, orig_w = image.shape[:2]
        r = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
        resize_size = (int(orig_w * r), int(orig_h * r))
        if r != 1:
            resized_img = cv2.resize(image, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = image

        # padding
        padded_img[:resized_img.shape[0], :resized_img.shape[1]] = resized_img
        
        # [H, W, C] -> [C, H, W]
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)


        return padded_img, r
