import random
import numpy as np
import argparse
from utils import *

IMG_H = int(218 * 0.4)
IMG_W = int(178 * 0.4)

"""
Display Mat

Selects a random image and displays its reconstruction along with the set of basis
images and weights.
"""
def load_checkpoints():
    # load w, h, and costs
    w = np.load('checkpoints/w_mat_final.npy')
    h = np.load('checkpoints/h_mat_final.npy')
    costs = np.load('checkpoints/costs_final.npy')

    return w, h, costs

def main(fp, num_imgs):
    # load celebA
    x = load_imgs(fp, num_imgs)

    # load mats
    w, h, costs = load_checkpoints()

    # sample random image to display
    print('Sampling image...')
    rand = random.randint(0, num_imgs)
    show_img(x[:, rand])

    # extract facial features
    print('Plotting basis vectors...')
    feats = []
    for i in range(w.shape[1]):
        feats.append(w[:, i].reshape([IMG_H, IMG_W]))

    plot_grid(feats, scale=True)

    # extract importance matrix for each image
    print('Plotting weights...')
    importance = []
    for i in range(h.shape[0]):
        val = h[i, rand]
        arr = np.ones(3072) * val
        importance.append(arr.reshape([32, 32, 3]))
    
    plot_grid(importance)

    # combine features and importance matrix to get estimate of image
    print('Plotting reconstructed image...')
    a = np.dot(w, h[:, rand]) * 255
    a = a.astype(np.uint8)
    show_img(a)


if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='nmf')
    parser.add_argument('--num_imgs', default=10000, type=int, help='Number of images to use.')
    parser.add_argument('--fp', type=str, help='Filepath of dataset. If using CelebA, point to "img_align_celeba" folder.')

    args = parser.parse_args()
    main(args.fp, args.num_imgs)
