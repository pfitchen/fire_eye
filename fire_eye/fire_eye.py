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
		pass


	def train_model(self):
		self.classifier = Classifier(pos_data_path=pos_data_path,
									 neg_data_path=neg_data_path)


	def evaluate_model(self):
		pass


	def save_model(self):
		pass


	def load_model(self, model_path:str):
		self.classifier = Classifier(pos_data_path=pos_data_path,
									 neg_data_path=neg_data_path,
									 model_path=model_path)