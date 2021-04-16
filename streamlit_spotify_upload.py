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

# Glob for reading file names in a folder
import glob
# json for storing data in json file
import json
import pandas as pd

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial
from spotify_algo import get_all_images
from spotify_algo import get_vector
from spotify_algo import match_id
from spotify_algo import cluster


st.set_option('deprecation.showfileUploaderEncoding', False)
images = {}
images = get_all_images()
nnn = []

uploaded_file = st.file_uploader("Upload an image...", type="jpg")
slid=st.slider('How many similar images you want to see?', 0, 5, 1)
if st.button('Submit'):
	if uploaded_file is not None:
		vector = get_vector(uploaded_file)
		nnn = cluster(vector)
		im = Image.open(uploaded_file)
		st.image(im, caption = "Uploaded image")
		slid = slid + 1
		st.write("Similar Images")
		for i in range(0, slid):
			for k in range(0, len(images)):
				if(nnn[i]['similar_pi'] == images[k]['product_id']):
					st.image(images[k]['image'])
					break
