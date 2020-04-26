# FireEye

FireEye is a toolset for exploring the feasibility of image based wildfire detection. However, it could easily be adapted for other image classification tasks. It's main advantage is the inclusion of a Google Images scraper for automatically generating loosely labelled datasets.

Fire Eye contains three low-level modules that are combined in a top-level module for ease of use in scripts or by other users. Each of these modules are implemented as classes to abstract away as much detail as possible. Unit tests are included for each module, but as always more testing would be better. The three low level modules are the Scraper, Preprocessor, and Classifier. The Scraper uses the Selenium package and a chromium webdriver for scraping Google Images. The Preprocessor takes as input a list of directories with images to consider positively labelled and another similar list of directories with images that should be negatively labelled. It then preprocesses all images in these directories into a final dataset ready for model training or evaluation. The Classifier contains all of the PyTorch implementation details for training, testing, and evaluating models. It’s also responsible for generating relevant figures. The top-level FireEye class manages implementations of each of the three sub-modules so that potential users only need to learn one simple API. The code should work out of the box for any binary image classification task and can even generate a new dataset from Google Images. Refer to the GitHub (https://github.com/pfitchen/fire_eye) for more details.


To use the tool, run one of the Make commands.

"make" should run the __main__.py demo application.

"make init" should install the required packages.

"make test" runs the small set of unit tests, but suppresses stdout.

"make test-verbose" runs the small set of unit tests without suppressing stdout.

"make clean" removes any compiled .pyc files within FireEye.

"make clean-scraped" removes any scraped images from the project.

"make clean-preprocessed" removes all preprocessed images. These are the images that are automatically considered part of the training/validation/test datasets, so it should be cleaned anytime a fresh dataset is desired!!!

"make clean-all" does all three of these cleaning tasks with a single command.