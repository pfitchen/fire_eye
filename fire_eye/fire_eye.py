from fire_eye.scraper.scraper import Scraper
from fire_eye.preprocessor.preprocessor import Preprocessor
from fire_eye.classifier.classifier import Classifier
from torchvision import datasets, models, transforms
import os


class FireEye():
	def __init__(self):
		self.download_path = os.path.join(os.getcwd(), 'data', 'scraped')
		self.preprocessed_path = os.path.join(os.getcwd(), 'data',
											  'preprocessed')
		self.pos_data_path = os.path.join(self.preprocessed_path, 'pos')
		self.neg_data_path = os.path.join(self.preprocessed_path, 'neg')
		self.models_path = os.path.join(os.getcwd(), 'data', 'models')

		transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
									transforms.RandomRotation(5),
									transforms.ColorJitter(brightness=0.05, 
														   contrast=0.05, 
														   saturation=0.05, 
														   hue=0.05),
									transforms.ToTensor()])
		self.transform = transform
		
		self.preprocessor = Preprocessor(preprocessed_path=self.preprocessed_path,
										 img_size=(32, 32))
		self.classifier = None


	def scrape_more_data(self, num_images:int, pos_queries:list=[], 
						 neg_queries:list=[]):
		queries = pos_queries + neg_queries
		with Scraper(headless=True) as s:
			for query in queries:
				s.scrape_google_images(num_images=num_images, query=query,
									   download_path=self.download_path)
		pos_paths = [os.path.join(self.download_path, pos_query)
					 for pos_query in pos_queries]
		neg_paths = [os.path.join(self.download_path, neg_query)
					 for neg_query in neg_queries]
		self.preprocessor.preprocess(pos_paths=pos_paths, neg_paths=neg_paths)


	def load_kaggle_data(self):
		kaggle_data_path = os.path.join(os.getcwd(), 'data', 'kaggle_data')
		fire_dataset_path = os.path.join(kaggle_data_path, 'fire_dataset')
		FireDetection_path = os.path.join(kaggle_data_path, 'Fire-Detection')
		pos_paths = [os.path.join(fire_dataset_path, 'fire_images'),
					 os.path.join(FireDetection_path, '1')]
		neg_paths = [os.path.join(fire_dataset_path, 'non_fire_images'),
					 os.path.join(FireDetection_path, '0')]
		self.preprocessor.preprocess(pos_paths=pos_paths, neg_paths=neg_paths)


	def get_dataset_sizes(self):
		total_size = len(os.listdir(self.pos_data_path)) +\
					 len(os.listdir(self.neg_data_path))
		return {'total':total_size,
				'train':int(round(0.6*total_size)),
				'val':int(round(0.2*total_size)),
				'test':int(round(0.2*total_size))}


	def train_model(self, num_epochs:int):
		if self.classifier is None:
			self.classifier = Classifier(pos_data_path=self.pos_data_path,
										 neg_data_path=self.neg_data_path,
										 transform = self.transform)
		return self.classifier.train(num_epochs=num_epochs)


	def evaluate_model(self):
		if self.classifier is None:
			self.classifier = Classifier(pos_data_path=self.pos_data_path,
										 neg_data_path=self.neg_data_path,
										 transform = self.transform)
		return self.classifier.evaluate()


	def save_model(self, fname:str):
		if not (fname.endswith('.pt') or fname.endswith('.pth')):
			fname = fname + '.pt'
		model_path = os.path.join(self.models_path, fname)

		if self.classifier is None:
			self.classifier = Classifier(pos_data_path=self.pos_data_path,
										 neg_data_path=self.neg_data_path,
										 transform = self.transform)
		self.classifier.save(model_path=model_path)


	def load_model(self, fname:str):
		if not (fname.endswith('.pt') or fname.endswith('.pth')):
			fname = fname + '.pt'
		model_path = os.path.join(self.models_path, fname)
		self.classifier = Classifier(pos_data_path=self.pos_data_path,
									 neg_data_path=self.neg_data_path,
									 model_path=model_path,
									 transform=self.transform)