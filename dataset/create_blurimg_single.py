import argparse
from tqdm import tqdm
import os
import numpy as np
import cv2
from blur import *
from utils import *

class CreateBlurImgSingle:
	'''
		class to create blur images for single image dataset
	'''
	def __init__(self, img_path, blur_method='defocus', calc='psnr'):
		'''
		create blurred image from the single image with random hyperparameters
		'''
		available = ['.jpg', '.png', '.JPG', '.PNG', 'JPEG']
		metrics = ['psnr', 'ssim', 'degree']

		assert os.path.splitext(img_path)[-1] in available, 'You cannot open this type of file'
		assert calc in metrics, 'You cannot use this metric'

		self.calc = calc
		self.blur_method = blur_method
		if blur_method == 'defocus':
			self.parameters = {'mean':50, 'var':20, 'dmin':10, 'dmax':100}
		elif blur_method == 'deblurGAN':
			self.parameters = {'canvas':64,
								'iters':2000,
								'max_len':60,
								'expl':np.random.choice([0.003, 0.001, 0.0007, 0.0005,
														0.0003, 0.0001]),
								'part':np.random.choice([1, 2, 3])
							  }
		else:
			raise ValueError("Not available metric yet")

		#generate and save samples randomly with one image sample
		self.img_path = img_path

	def generate_samples_randomly(self, result_pth=None,
								  sample_num=2000, save=True):
		if result_pth is None:
			os.makedirs('../examples/'+self.calc, exist_ok=True)
			result_pth = '../examples/'+self.calc
		
		if os.path.isfile(os.path.join(result_pth, 'results.txt')):
			os.remove(os.path.join(result_pth, 'results.txt'))

		print("Generate samples...")
		for iteration in tqdm(range(sample_num)):
			if self.blur_method == 'defocus':
				image = cv2.imread(self.img_path)
				blurred, degree = blurring(image, self.parameters)
				if save:
					# save image with name contain degree information
					if self.calc == 'psnr':
						degree = psnr(image, blurred)
					elif self.calc == 'ssim':
						degree = ssim(image, blurred)

					path = os.path.join(result_pth, f'sample_{iteration}'+os.path.splitext(self.img_path)[-1])
					cv2.imwrite(path, blurred)

					with open(os.path.join(result_pth, 'results.txt'), 'a') as f:
						f.write(str(degree)+'\n')

			elif self.blur_method == 'deblurGAN':
				self.parameters['expl'] = np.random.choice([0.003, 0.001, 0.0007, 0.0005, 0.0003, 0.0001])
				self.parameters['part'] = np.random.choice([1, 2, 3])
				trajectory = Trajectory(self.parameters).fit()
				psf, degree = PSF(self.parameters['canvas'], trajectory = trajectory).fit()
				image, blurred = BlurImage(self.img_path, psf, self.parameters['part']).blur_image()
				if save:
					# save image with name contain degree information
					if self.calc == 'psnr':
						degree = psnr(image, blurred)
					elif self.calc == 'ssim':
						degree = ssim(image, blurred)
					path = os.path.join(result_pth, 'sample_{iteration}'+os.path.splitext(self.img_path)[-1])
					cv2.imwrite(path, blurred)

					with open(os.path.join(result_pth, 'results.txt'), 'a') as f:
						f.write(str(degree)+'\n')
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='This program generates samples for one image sample')
	parser.add_argument("--sample", type=str, help="You should enter the image path", default=None)
	parser.add_argument("--blur", type=str, help="defocus, deblurGAN is available", default='defocus')
	parser.add_argument("--calc", type=str, help="psnr, ssim, degree is available", default='degree')
	parser.add_argument("--iter", type=int, help="Iteration(sample number) to create blur samples", default=2000)
	args = parser.parse_args()
	
	assert args.sample != None, 'Program needs the sample name'
	blurrer = CreateBlurImgSingle(args.sample, args.blur, args.calc)
	blurrer.generate_samples_randomly(sample_num = args.iter)
