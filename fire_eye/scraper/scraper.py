'''
NOTE:	USE IN A CONTEXT MANAGER!!!
		There seems to be an issue with webdriver.quit() being called
		when the program is ending. So this class is intended to be used
		in a context manager instead to ensure that the webdriver is always
		properly closed without issue.

NOTE:   It's quite possible that either the webdriver falls out of date
		or that some of the css_selectors change as these sites are
		possibly updated. It's recommended to run the test module to
		see if any manual updates are needed.
'''


import os
import io
import requests
import urllib
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options


class Scraper():


	# THIS CLASS SHOULD BE USED IN A CONTEXT MANAGER!!!
	# create and initialize a chrome webdriver object
	def __init__(self, driver_path:str=None, headless:bool=True):
		if driver_path is None:
			driver_path = os.path.join(os.path.dirname(__file__), 'drivers', 'chromedriver')
		if not os.path.isfile(driver_path):
			raise ValueError(f'{driver_path} is not a valid file path.')
		opts = Options()
		opts.headless = headless
		self.wd = Chrome(executable_path=driver_path, options=opts)


	# BUT USE THIS CLASS IN A CONTEXT MANAGER WITH __EXIT__!!!
	# quit the webdriver on object destruction
	# (this will produce an error if program is also ending, 
	# so use a context manager!!!)
	def __del__(self):
		self.wd.quit()


	# context manager magic enter method, just return self
	def __enter__(self):
		return self


	# context manager magic exit method, just call destructor
	# (the destructor quits webdriver)
	def __exit__(self, type, value, traceback):
		self.__del__()


	# enter a query and search on Google. update webdriver
	def search_google(self, query:str):
		self.wd.get('https://google.com')
		search_box = self.wd.find_element_by_css_selector('input.gLFyf')
		search_box.send_keys(query)
		search_box.submit()


	# enter a query and search on Google Images. update webdriver
	def search_google_images(self, query:str):
		self.wd.get('https://www.google.com/imghp?hl=en')
		search_box_selector = '#sbtc > div > div.a4bIc > input'
		search_box = self.wd.find_element_by_css_selector(search_box_selector)
		search_box.send_keys(query)
		search_box.submit()


	# enter a query to search on Google Images and download num_images to a 
	# target directory. defaults to <cwd>/data/<query> if no directory is 
	# specified. update webdriver
	def scrape_google_images(self, num_images:int, query:str,
							 download_path:str):
		# helper function, scroll to bottom of page to load more images
		def scroll_to_bottom():
			script = 'window.scrollTo(0, document.body.scrollHeight);'
			self.wd.execute_script(script)

		# helper function, attempt to download a target image from a url.
		# don't terminate on exceptions. return 1 if successful, 0 otherwise.
		def download_img():
			try:
				# img_content = requests.get(img_url).content
				resp = urllib.request.urlopen(img_url)
				img = np.asarray(bytearray(resp.read()), dtype="uint8")
				img = cv2.imdecode(img, cv2.IMREAD_COLOR)
				# temp_file = io.BytesIO(img_content)
				# img = Image.open(temp_file).convert('RGB')
				img_fname = f'{query.replace(" ", "_")}_{img_count}.png'
				img_path = os.path.join(download_path, img_fname)
				cv2.imwrite(img_path, img)
				# with open(img_path, 'wb') as f:
				# 	img.save(f, quality=85)
				return 1
			except:
				return 0
		
		# check download_path
		if not os.path.isdir(download_path):
			raise ValueError(f'{download_path} is not a valid path.')
		download_path = os.path.join(download_path, f'{query}')
		if not os.path.isdir(download_path):
			os.mkdir(download_path)

		# this is the same as search_google_images(), 
		# but I want this fn to standalone.
		self.wd.get('https://www.google.com/imghp?hl=en')
		search_box_selector = '#sbtc > div > div.a4bIc > input'
		search_box = self.wd.find_element_by_css_selector(search_box_selector)
		search_box.send_keys(query)
		search_box.submit()

		# click on an image and look for downloadable urls. attempt to download 
		# each url until num_images have successfully been downloaded.
		img_count = 0
		img_urls = set() # to avoid repeat urls
		prog_bar = tqdm(total=num_images)
		while img_count < num_images:
			# load more images if needed
			scroll_to_bottom()
			img_elems = self.wd.find_elements_by_css_selector('img.Q4LuWd')

			# try clicking on each possible image element,
			# then try to download each possible image url
			for img_elem in img_elems:
				# if an issue occurs, move on to next possible img element
				try:
					img_elem.click()
				except:
					continue
				poss_urls = self.wd.find_elements_by_css_selector('img.n3VNCb')

				# try to download each possible img url
				for poss_url in poss_urls:
					poss_url_src = poss_url.get_attribute('src')
					if poss_url_src and 'http' in poss_url_src:
						img_url = poss_url.get_attribute('src')
						if img_url not in img_urls:
							ret_val = download_img()
							img_count += ret_val
							img_urls.add(img_url) # avoid repeated urls
							prog_bar.update(ret_val)
						if img_count >= num_images:
							prog_bar.close()
							return


# scrape 100 Fire and NoFire images of a forest
if __name__ == '__main__':
	download_path = os.path.join(os.path.getcwd(), 'data', 'test_scraper_data')
	with Scraper(headless=True) as scrpr:
		scrpr.scrape_google_images(num_images=20, query='Forest Fire')
		scrpr.scrape_google_images(num_images=20, query='Forest')

