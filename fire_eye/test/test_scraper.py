import unittest
from fire_eye.scraper.scraper import Scraper
import os
import shutil


class TestScraper(unittest.TestCase):
	# create a scraper object for all tests (faster than per test)
	# recall Scraper should be used in a context manager...
	@classmethod
	def setUpClass(cls):
		cls.scrpr = Scraper(headless=True).__enter__()


	# scraper object is used in a context manager,
	# so call exit method
	@classmethod
	def tearDownClass(cls):
		cls.scrpr.__exit__(None, None, None)


	def test_search_google(self):
		# just checking that this code runs...
		try:
			self.scrpr.search_google(query='COVID-19')
		except:
			self.fail(f'An error occurred searching Google... - {e}')


	def test_search_google_images(self):
		# check that this code runs...
		try:
			self.scrpr.search_google_images(query='Dogs')
		except:
			self.fail(f'An error occurred searching Google Images... - {e}')
		# check that 100 results are shared on first page
		# (indicates that correct page is loaded)
		img_elems = self.scrpr.wd.find_elements_by_css_selector('img.Q4LuWd')
		self.assertEqual(len(img_elems), 100)


	def test_scrape_google_images(self):
		# set the query, target num_images, and download_dir for the test
		query = 'dogs'
		num_images = 5
		download_path = os.path.join(os.getcwd(), 'data', 'test_scraper_data')
		imgs_path = os.path.join(download_path, f'{query}')

		# check that code runs...
		try:
			self.scrpr.scrape_google_images(num_images=num_images, query=query,
											download_path=download_path)
		except Exception as e:
			self.fail(f'An error occurred scraping Google Images... - {e}')
		# check that the correct number of images were downloaded and that the
		# download dir was properly created if needed
		num_files = len(os.listdir(imgs_path))
		self.assertEqual(num_files, num_images)

		# remove downloaded images to end test
		shutil.rmtree(imgs_path)


# run all tests
if __name__ == '__main__':
	unittest.main()