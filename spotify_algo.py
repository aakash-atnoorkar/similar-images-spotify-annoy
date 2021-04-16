import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
# For saving 'feature vectors' into a txt file
import numpy as np
# Glob for reading file names in a folder
import glob
import os.path
import io
from imageio import imread
import base64
from io import BytesIO
import PIL
from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
from PIL import Image
import time
from skimage import io
# json for storing data in json file
import json
import pandas as pd
from tqdm import tqdm
import ntpath
import cv2

from sklearn.metrics.pairwise import cosine_similarity
import scipy as sc
from scipy import spatial
import matplotlib.pyplot as plt


# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial
st.set_option('deprecation.showfileUploaderEncoding', False)

#image_paths = glob.glob(image_path+'*.jpg')
image_paths = ['https://similariy-search-images.s3-us-west-2.amazonaws.com/images/0_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/997_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/875_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/997_1.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/178_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/128_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/100012_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/4008632_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/13219_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/1164_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/1674056_0.jpg', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/images/4672_0.jpg']


allfiles = ['https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/0_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/997_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/875_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/997_1.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/178_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/128_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/100012_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/4008632_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/13219_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/1164_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/1674056_0.jpg.npz', 'https://similariy-search-images.s3-us-west-2.amazonaws.com/imageScraped/4672_0.jpg.npz']
st.title("Visual Search")
def match_id(filename):
    product_id = '_'.join(filename.split('_')[:-1])
    return product_id

def load_image(image):

	#image = image.decode('utf-8')
	image = np.array(image.read())
	img = tf.io.decode_jpeg(image, channels=3)
	#image = img_to_array(image)
	# reshape data for the model
	img = tf.image.resize_with_pad(img, 224, 224)
	# Converts the data type of uint8 to float32 by adding a new axis
	# img becomes 1 x 224 x 224 x 3 tensor with data type of float32
	# This is required for the mobilenet model we are using
	img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
	return img


def cluster(v0):
	#st.write("-------Annoy Index Generation----------------")
    # Defining data structures as empty dict
	file_index_to_file_name = {}
	file_index_to_file_vector = {}
	file_index_to_product_id = {}
	# Configuring annoy parameters
	dims = 1792
	n_nearest_neighbors = 20
	trees = 10000
	# Reads all file names which stores feature vectors

	t = AnnoyIndex(dims, metric='angular')
	j = 0
	for file_index, i in enumerate(allfiles):


    # Reads feature vectors and assigns them into the file_vector
		file_vector = np.loadtxt(i)

		# Assigns file_name, feature_vectors and corresponding product_id
		file_name = os.path.basename(i).split('.')[0]
		file_index_to_file_name[file_index] = file_name
		file_index_to_file_vector[file_index] = file_vector
		file_index_to_product_id[file_index] = match_id(file_name)

		# Adds image feature vectors into annoy index
		t.add_item(file_index, file_vector)

		print("---------------------------------")
		print("Annoy index     : %s" %file_index)
		print("Image file name : %s" %file_name)
		print("Product id      : %s" %file_index_to_product_id[file_index])
		j = j + 1
		# st.write("Printing j")
        #print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
	file_index_to_file_name[j] = 'new_file'
	file_index_to_file_vector[j] = v0
	file_index_to_product_id[j] = 'new_file'

	t.add_item(j, v0)
	#Builds
	# Builds annoy index
	t.build(trees)

	print ("Step.1 - ANNOY index generation - Finished")
	print ("Step.2 - Similarity score calculation - Started ")

	global named_nearest_neighbors
	named_nearest_neighbors = []

	master_file_name = file_index_to_file_name[j]
	master_vector = file_index_to_file_vector[j]
	master_product_id = file_index_to_product_id[j]

	nearest_neighbors = t.get_nns_by_item(j, n_nearest_neighbors)

	for k in nearest_neighbors:

		# Assigns file_name, image feature vectors and product id values of the similar item
		neighbor_file_name = file_index_to_file_name[k]
		neighbor_file_vector = file_index_to_file_vector[k]
		neighbor_product_id = file_index_to_product_id[k]

		# Calculates the similarity score of the similar item
		similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
		rounded_similarity = int((similarity * 10000)) / 10000.0

		# Appends master product id with the similarity score
		# and the product id of the similar items
		named_nearest_neighbors.append({
		'similarity': rounded_similarity,
		'master_pi': master_product_id,
		'similar_pi': neighbor_product_id})

	#print(named_nearest_neighbors)
	return named_nearest_neighbors

def get_vector(image):
    x= []
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
	# Loads the module
    module = hub.load(module_handle)

	#Preprocess the image
    img = load_image(image)
        # Calculate the image feature vector of the img
    features = module(img)
        # Remove single-dimensional entries from the 'features' array
    feature_set = np.squeeze(features)
    return feature_set

def get_all_images():

    #print(f'Found [{len(image_paths)}] images')
    images = []
    for image_path in image_paths:
        #image = cv2.imread('https://similariy-search-images.s3-us-west-2.amazonaws.com/images/100012_0.jpg', 3)
        #st.write(image)
        #b,g,r = cv2.split(image)           # get b, g, r
        image = io.imread(image_path)         # switch it to r, g, b
        image = cv2.resize(image, (200, 200))
        product_id = match_id(ntpath.basename(image_path))
        images.append({
		  'image_path' : ntpath.basename(image_path),
		    'image' : image,
		   'product_id':product_id
		})
    return images
