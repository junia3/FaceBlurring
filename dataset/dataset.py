import argparse
import pandas as pd
import numpy as np
import math, cv2, os
from torch.utils.data import Dataset
from utils import *
# test용
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
	def __init__(self, txt_file, option='clean', transform=None):
		'''
			face dataset module
			txt file must include root directory of sample images
		'''
		with open(txt_file, 'r') as f:
			lines = f.readlines()
			sample_root = [l.rstrip('\n') for l in lines]
	
		if option=='clean':
			self.sample_paths = self._get_clean_samples(sample_root)
			self.labels = 100*np.ones(len(self.sample_paths))

		elif option == 'blur':
			self.sample_paths, self.labels = self._get_blur_samples(sample_root)
		else:
			raise ValueError("option should be 'clean' or 'blur'")

	def _get_clean_samples(self, roots):
		'''
			Inner function to get all clean samples under sample root
			This function only return clean images
		'''
		paths = []
		for root in roots:
			for (path, directory, files) in os.walk(root):
				for filename in files:
					ext = os.path.splitext(filename)[-1]
					if ext in ['.jpg', '.png'] and 'clean' in path:
						paths += [os.path.join(path. filename)]
		return paths

	def _get_blur_samples(self, roots):
		'''
			Inner function to get all blur samples under sample root
			This function only return blur images
		'''
		paths = []
		labels = []
		label_path = "../data/label/label.csv"
		assert os.path.isfile(label_path), "label file does not exist"

		df = pd.read_csv(label_path)

		for root in roots:
			for (path, directory, files) in os.walk(root):
				for filename in files:
					ext = os.path.splitext(filename)[-1]
					if ext in ['.jpg', '.png'] and 'blur' in path:
						filepath = os.path.join(path, filename)
						paths += [filepath]
						labels.append(df.loc[df['filename'] == filepath]['label'].item())
						
		return paths, labels


	def __len__(self):
		return len(self.sample_paths)

	def __getitem__(self, idx):
		img_path, label = self.sample_paths[idx], self.labels[idx]
		image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
		return image, label

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='This program makes dataset for face samples')
	parser.add_argument('--source', type=str, help='Needs txt file containing roots', default = '../config/test.txt')
	parser.add_argument('--option', type=str, help='choose between clean and blur', default='blur')
	args = parser.parse_args()
	dataset = FaceDataset(args.source, args.option)

	plt.figure(figsize=(100, 100))
	for i in range(len(dataset)):
		image, label = dataset[i]
		plt.subplot(5, 5, i+1)
		plt.imshow(image)
		plt.title(f'PSNR : {label:.2f}', fontsize=80)
		if i==24:
			break
	plt.savefig('result.png')
