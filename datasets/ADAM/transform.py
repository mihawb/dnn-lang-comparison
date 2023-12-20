import pandas as pd
import numpy as np
from PIL import Image
import torchvision


class Resize:
	'''Resize the image and convert the label
		to the new shape of the image'''
	def __init__(self, new_size=(256, 256)):
		self.new_width = new_size[0]
		self.new_height = new_size[1]

	def __call__(self, image_label_sample):
		image = image_label_sample[0]
		label = image_label_sample[1]
		c_x, c_y = label
		original_width, original_height = image.size
		image_new = torchvision.transforms.functional.resize(image, (self.new_width, self.new_height))
		c_x_new = c_x * self.new_width /original_width
		c_y_new = c_y * self.new_height / original_height
		return image_new, (c_x_new, c_y_new)


class RandomHorizontalFlip:
	'''Horizontal flip the image with probability p.
		Adjust the label accordingly'''
	def __init__(self, p=0.5):
		if not 0 <= p <= 1:
			raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
		self.p = p  # float between 0 to 1 represents the probability of flipping

	def __call__(self, image_label_sample):
		image = image_label_sample[0]
		label = image_label_sample[1]
		w, h = image.size
		c_x, c_y = label
		if np.random.random() < self.p:
			image = torchvision.transforms.functional.hflip(image)
			label = w - c_x, c_y
		return image, label


class RandomVerticalFlip:
	'''Vertically flip the image with probability p.
		Adjust the label accordingly'''
	def __init__(self, p=0.5):
		if not 0 <= p <= 1:
			raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
		self.p = p  # float between 0 to 1 represents the probability of flipping

	def __call__(self, image_label_sample):
		image = image_label_sample[0]
		label = image_label_sample[1]
		w, h = image.size
		c_x, c_y = label
		if np.random.random() < self.p:
			image = torchvision.transforms.functional.vflip(image)
			label = c_x, h - c_y
		return image, label


class RandomTranslation:
	'''Translate the image by randomaly amount inside a range of values.
		Translate the label accordingly'''
	def __init__(self, max_translation=(0.2, 0.2)):
		if (not 0 <= max_translation[0] <= 1) or (not 0 <= max_translation[1] <= 1):
			raise ValueError(f'Variable max_translation should be float between 0 to 1')
		self.max_translation_x = max_translation[0]
		self.max_translation_y = max_translation[1]

	def __call__(self, image_label_sample):
		image = image_label_sample[0]
		label = image_label_sample[1]
		w, h = image.size
		c_x, c_y = label
		x_translate = int(np.random.uniform(-self.max_translation_x, self.max_translation_x) * w)
		y_translate = int(np.random.uniform(-self.max_translation_y, self.max_translation_y) * h)
		image = torchvision.transforms.functional.affine(image, translate=(x_translate, y_translate), angle=0, scale=1, shear=0)
		label = c_x + x_translate, c_y + y_translate
		return image, label


class ImageAdjustment:
	'''Change the brightness and contrast of the image and apply Gamma correction.
		No need to change the label.'''
	def __init__(self, p=0.5, brightness_factor=0.8, contrast_factor=0.8, gamma_factor=0.4):
		if not 0 <= p <= 1:
			raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
		self.p = p
		self.brightness_factor = brightness_factor
		self.contrast_factor = contrast_factor
		self.gamma_factor = gamma_factor

	def __call__(self, image_label_sample):
		image = image_label_sample[0]
		label = image_label_sample[1]

		if np.random.random() < self.p:
			brightness_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
			image = torchvision.transforms.functional.adjust_brightness(image, brightness_factor)

		if np.random.random() < self.p:
			contrast_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
			image = torchvision.transforms.functional.adjust_contrast(image, contrast_factor)

		if np.random.random() < self.p:
			gamma_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
			image = torchvision.transforms.functional.adjust_gamma(image, gamma_factor)

		return image, label
	

def load_image(df, idx):
	image_name = df.loc[idx, 'imgName']
	data_type = 'AMD' if image_name.startswith('A') else 'Non-AMD'
	image_path = f'./Training400/{data_type}/{image_name}'
	image = Image.open(image_path)
	bbox = (df.loc[idx, 'Fovea_X'], df.loc[idx, 'Fovea_Y'])
	return image, bbox


def progress_bar(curr_img, max_img, bar_length=30):
	percent = curr_img / max_img
	hashes = '#' * int(round(percent * bar_length))
	spaces = ' ' * (bar_length - len(hashes))
	print(f'\rAugumenting images: [{hashes}{spaces}] {curr_img}/{max_img} ({percent*100:.2f}%)', end='')


def main():
	fovea_df = pd.read_excel('./Training400/Fovea_location.xlsx', index_col='ID')
	fovea_df = fovea_df[(fovea_df[['Fovea_X', 'Fovea_Y']] != 0).all(axis=1)]

	augumentations = [RandomHorizontalFlip(), RandomVerticalFlip(), RandomTranslation(), ImageAdjustment()]
	resize = Resize()

	new_fovea = {col: [] for col in fovea_df}

	for idx in fovea_df.index:
		progress_bar(idx+1, len(fovea_df))
		img, bbox = load_image(fovea_df, idx)
		imgName = fovea_df.loc[idx, 'imgName']
		imgName = imgName[:imgName.rfind('.')]
		imgNames = [imgName + '.jpg', imgName + '-1.jpg', imgName + '-2.jpg']

		a1, a2 = np.random.choice(augumentations, size=2, replace=False)
		img1, bbox1 = a1((img, bbox))
		img2, bbox2 = a2((img, bbox))

		img, bbox = resize((img, bbox))
		img1, bbox1 = resize((img1, bbox1))
		img2, bbox2 = resize((img2, bbox2))

		for image, name in zip([img, img1, img2], imgNames):
			data_type = 'AMD' if name.startswith('A') else 'Non-AMD'
			image.save(f'./Training1200/{data_type}/{name}')

		new_fovea['imgName'].extend(imgNames)
		new_fovea['Fovea_X'].extend([bbox[0], bbox1[0], bbox2[0]])
		new_fovea['Fovea_Y'].extend([bbox[1], bbox1[1], bbox2[1]])

	pd.DataFrame(new_fovea).reset_index(names='ID').to_csv('./Training1200/fovea_location.csv', index=False)


if __name__ == '__main__':
	main()