import random
import argparse
import numpy as np
from utils import *

"""
NMF Solver

Factorizes input matrix (images) into basis images (W) and their weights (H).
"""
class NMF_Solver(object):

	def __init__(self, x, k=4):
		# x should have shape (row, col) = (IMG_H*IMG_W, len(files))
		self.x = x

		# init W (m x k) and H (k x n)
		m, n = x.shape[0], x.shape[1]
		self.w = np.random.rand(m, k)
		self.h = np.random.rand(k, n)
		self.costs = []

		print('x shape = {}'.format(self.x.shape))
		print('w shape = {}'.format(self.w.shape))
		print('h shape = {}'.format(self.h.shape))

	def solve(self, itrs):
		for i in range(itrs):
			# update W and H sequentially
			self.w = self.update_w()
			self.h = self.update_h()

			# calculate cost (EUCLIDEAN DISTANCE between imgs and HW)
			cost = (np.linalg.norm(self.x - np.linalg.multi_dot([self.w, self.h])))**2
			self.costs.append(cost)
			print('Iteration {}: cost = {}'.format(i, cost))

			# save checkpoint
			if i > 49 and i % 50 == 0:
				self.save_checkpoint(i)

		# save final checkpoint
		self.save_checkpoint('final')

	def update_w(self):
		# W .* (X * H')
		top = self.w * np.linalg.multi_dot([self.x, np.transpose(self.h)])

		# W * (H * H')
		bot = np.linalg.multi_dot([self.w, np.linalg.multi_dot([self.h, np.transpose(self.h)])])
		
		return top / bot

	def update_h(self):
		# H .* (W' * X)
		top = self.h * np.linalg.multi_dot([np.transpose(self.w), self.x])

		# (W * W') * H
		bot = np.linalg.multi_dot([np.linalg.multi_dot([np.transpose(self.w), self.w]), self.h]) 

		return top / bot

	def save_checkpoint(self, itr):
		# save W, H matrices and costs up to this point
		print('saving checkpoint!')
		np.save('checkpoints/w_mat_{}'.format(itr), self.w, allow_pickle=True)
		np.save('checkpoints/h_mat_{}'.format(itr), self.h, allow_pickle=True)
		np.save('checkpoints/costs_{}'.format(itr), np.asarray(self.costs), allow_pickle=True)

		if itr == 'final':
			# save X matrix at very end
			np.save('checkpoints/x_mat_{}'.format(itr), self.x, allow_pickle=True)


def main(args):
	# load CelebA
	# fp = '/Users/nikogrupen/Documents/developer/playground/data/img_align_celeba'
	imgs = load_imgs(args.fp, args.num_imgs)

	# sample random image to display
	rand = random.randint(0, args.num_imgs)
	show_img(imgs[:, rand])

	# init NMF
	nmf = NMF_Solver(imgs, k=args.k)

	# solve
	nmf.solve(args.itrs)


if __name__  == '__main__':
	parser = argparse.ArgumentParser(description='nmf')
	parser.add_argument('--num_imgs', default=10000, type=int, help='Number of images to use.')
	parser.add_argument('--itrs', default=1000, type=int, help='Number of iterations.')
	parser.add_argument('--k', default=1000, type=int, help='Size of factorization rank.')
	parser.add_argument('--fp', type=str, help='Filepath of dataset. If using CelebA, point to "img_align_celeba" folder.')

	args = parser.parse_args()
	main(args)