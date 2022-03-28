import cv2
import numpy as np


def create_images(images, labels):
    single_image_height = images.shape[1]
    single_image_width = images.shape[2]
    number_of_images = images.shape[0]
    height = single_image_height+4+30
    width = (2+single_image_width)*number_of_images+2
    image = np.ones((height, width, 1), np.uint8)*255
    for i, test_image in enumerate(np.squeeze(images[:, ...])):
        x = 2+i*(single_image_width+2)
        y = 2
        image[y:y+single_image_height, x:x+single_image_width, 0] = test_image[:, :]*255
        cv2.putText(
            image, str(labels[i]),
            (x+5, y+single_image_height+single_image_height-2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return image
