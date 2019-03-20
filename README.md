# nmf

Implementation of non-negative matrix factorization via multiplicative updates. 

Currently assumes use of CelebA dataset, update IMG_H and IMG_W globals for other datasets.

__nmf.py__

Runs non-negative matrix factorization according to the following command line arguments:
- num_imgs = number of input images to use from dataset.
- itrs = number of iterations.
- k = factorization rank. Note: larger k pays more accurate reconstructions with longer compute times.
- fp = filepath to your dataset. For the CelebA dataset, point this to the "img_align_celeba" folder.

Example Run:
`python nmf.py --num_imgs 4000 --itrs 1000 --k 99 --fp ../data/img_align_celeba`

__display_mat.py__

Displays a sample image, basis vectors, weights, and image reconstructions. The following commands are available:
- num_imgs = number of input images to use from dataset.
- fp = filepath to your dataset. For the CelebA dataset, point this to the "img_align_celeba" folder.

Example Run:
`python display_mat.py --num_imgs 4000 --fp ../data/img_align_celeba`

__References__
1.  D. D. Lee and H. S. Seung, [“Algorithms for non-negative matrix factorization”](https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)

2. N. Gillis,  [“The Why and How of Nonnegative Matrix Factorization”](https://arxiv.org/abs/1401.5226)


