import os
from scipy import misc
import cv2 
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


IMG_H = int(218 * 0.4)
IMG_W = int(178 * 0.4)

"""
Image utils
"""
def load_imgs(fp, num):
	# load images into numpy arrays
	files = os.listdir(fp)[:num]

	imgs = []	
	for f in sorted(files):
		print('working on image {}/{}'.format(len(imgs), len(files)))
		img = misc.imread(os.path.join(fp, f))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# shrink by half
		img = cv2.resize(img, (0,0), None, 0.4, 0.4)

		flat = img.reshape(img.shape[0] * img.shape[1]) / 255
		imgs.append(flat)

	# stack into matrix where each col is an img vector
	mat = np.stack(imgs, axis=1)

	return mat


def show_img(img):
    # CelebA images are (218, 178), but we are decreasing size to 0.4 of that for performance
    img = img.reshape([IMG_H, IMG_W])

    plt.imshow(img, cmap='gray')
    plt.show()


def plot_grid(imgs, cols=5, scale=False):
    n_imgs = len(imgs)

    fig = plt.figure(figsize=(10,10))
    
    for n, img in enumerate(imgs):
        a = fig.add_subplot(cols, np.ceil(n_imgs/float(cols)), n + 1)
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
        plt.imshow(img, cmap='gray')
        
    if scale:
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_imgs)
    
    plt.show()


