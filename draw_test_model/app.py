from flask import Flask, render_template, request,  url_for, flash, redirect
from werkzeug.exceptions import abort
import tensorflow as tf
import numpy as np
import random

import torch
from torch import nn

import urllib.request
import datetime

from os import listdir
from os.path import isfile, join

from PIL import Image

app = Flask(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


########## MODIFY THIS TO CHANGE THE MODELS APPEARING ##########

digitsModels = ["digits.conv3.nll.adam.20.20240304033700.pth", "digits.conv2.nll.adam.20.20240304030352.pth", "digits.conv2.nll.adam.20.20240304024502.pth", "digits.conv2.nll.adam.50.pth", "digits.conv2.nll.adam.30.pth", "digits.conv2.nll.100.keras"]
fashionModels = ["fashion.conv2.nll.adam.20.20240229225014.pth", "fashion.dense2.keras", "fashion.conv2.keras"]

digitsKeys = ("0", "1", "2","3","4","5","6","7","8","9")
fashionKeys = ("T-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot")

########## USEFUL FUNCTIONS ##########

def load_model(name: str):
	ext = name.split(".")[-1]
	# TensorFlow
	if ext == "keras":
		return tf.keras.models.load_model("models/" + name)
	elif ext == "pth":
		m = torch.jit.load("models/" + name)
		m.eval()					# Set to evaluation mode
		return m
	else:
		return None

def get_predict(model, pixels: np.ndarray, keys: tuple):
	if isinstance(model, nn.Module):
		# The extra bracket adds the channel and batch size 1, and the model expects a rescaled version
		prediction = torch.nn.Softmax()(model(torch.tensor(np.reshape(pixels, (1,1) + pixels.shape),dtype=torch.float) / 255.)).detach().numpy()[0]
	elif isinstance(model, tf.keras.models.Sequential):
		# The extra bracket makes it batch size 1, and the model expects non-scaled and no channels
		prediction = model(tf.convert_to_tensor([pixels])).numpy()[0]
	else:
		return None
	#pred = {'name': s}
	# ext = s.split(".")[-1]
	# # TensorFlow
	# if ext == "keras":
	# 	m = tf.keras.models.load_model("models/" + s)
	# 	prediction = m(tf.convert_to_tensor([pixels])).numpy()[0]		
	# # PyTorch
	# elif ext == "pth":
	# 	m = torch.jit.load("models/" + s)
	# 	m.eval()					# Set to evaluation mode
	# 	prediction = torch.nn.Softmax()(m(torch.tensor(np.reshape(pixels, (1,1) + pixels.shape),dtype=torch.float) / 255.)).detach().numpy()[0]	# The extra bracket adds the channel and batch size 1, and the model expects a rescaled version
	# else:
	# 	return render_template('index.html', notice="Unrecognized file extension.")


	pred = {}
	# Find the most likely prediction and its probability
	maxIndex = 0
	pred['probs'] = {}
	for i in range(0, len(prediction)):
		if prediction[maxIndex] < prediction[i]:
			maxIndex = i
		pred['probs'][keys[i]] = round(prediction[i] * 100, 2)
	pred['prob'] = round(prediction[maxIndex] * 100, 2)

	# Find the corresponding user-readable string corresponding to the most likely prediction
	pred['str'] = keys[maxIndex]
	
	return pred


########## THE MAIN APP ##########

@app.route('/', methods=('GET', 'POST'))

def index():
	if request.method == 'POST':

		if request.form['action'] == "saveimg":
			imgdata = request.form['oldimg']
			imgval =  request.form['oldimgval']
			type = request.form['type']
			if type == "digits":
				keys = digitsKeys
			elif type == "fashion":
				keys = fashionKeys
			else:
				return render_template('index.html', notice="Unknown type, not saved.")
			if imgval != '-':
				response = urllib.request.urlopen(imgdata)
				with open(f"static/savedtests/{type}.{imgval}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg", 'wb') as f:
					f.write(response.file.read())
			return render_template('index.html', notice=f"Saved a {keys[int(imgval)]}!" if imgval != '-' else "No value selected, not saved.")
		
		elif request.form['action'] == "detect":

			# Get pixels from the form in string form, then stick it in a (420, 420) array
			pixelstext = request.form['pixels']
			bigpixels = np.array([[0 for i in range(0, 420)] for j in range(0, 420)])
			for i in range(0, 420):
				for j in range(0, 420):
					bigpixels[i][j] = pixelstext[(j % 420) + (i * 420)]

			# Reduce size to (28, 28), value an integer in [0, 256)
			pixels = np.array([[0. for i in range(0, 28)] for j in range(0, 28)])
			imageData = ""
			for i in range(0, 28):
				for j in range(0, 28):
					darkness = int((np.ones(15) @ bigpixels[15*i:15*(i+1),15*j:15*(j+1)] @ np.ones(15)) * (17.0/15.0))		# divide 15 * 15 = 225 to average, then multiply 255, truncate, is already negative image
					# A little tweak on the input image to get it darker
					darkness = min(255, int(darkness * 1.5))
					pixels[i][j] = darkness
					imageData += "0, 0, 0, " + str(darkness) + ","
			imageData = imageData[0:-1] 	# Get rid of the last comma
			
			# What are we trying to classify?
			if request.form['type'] == "digits":
				textkeys = digitsKeys
				names = digitsModels
			elif request.form['type'] == "fashion":
				textkeys = fashionKeys
				names = fashionModels
			else:
				return render_template('index.html', notice="Something went wrong and you somehow requested a model we don't have.")
			
			# Load models
			preds = []
			for s in names:
				m = load_model(s)
				if m == None:
					return render_template('index.html', notice="Unrecognized file extension.")
				pred = get_predict(m, pixels, textkeys)
				if pred == None:
					return render_template('index.html', notice="Error getting model prediction.")
				pred['name'] = s
				preds.append(pred)
			
			return render_template('index.html', notice="Results:", results=preds, image=imageData, type=request.form['type'])
		else:
			return render_template('index.html', notice="Invalid POST request.")
	else:
		return render_template('index.html', notice="Draw a digit and see if the models can recognize it.  You can try drawing an article of clothing too (not so serious).")


@app.route('/tests', methods=('GET',))

def tests():

	# Get the list of files
	filelists = [[] for i in range(10)]
	for f in listdir('static/savedtests/'):
		parts = f.split('.')
		if parts[0] != 'digits':
			continue
		try:
			dig = int(parts[1])
		except:
			continue

		filelists[dig].append(join('static/savedtests/', f))

	#if request.method == 'GET':
	#	return render_template('tests.html', files=filelists)
	if request.method == 'GET':
		# Load the models
		models = {}
		for s in digitsModels:
			models[s] = load_model(s)

		# Check on the files
		preds = []
		for d in filelists:
			for f in d: 
				img = Image.open(f)
				data = {'image': f}
				img_tensor = np.asarray(img) @ np.array([1,1,1,1])
				pred = {}
				for s in models.keys():
					pred[s] = get_predict(models[s], img_tensor, digitsKeys)['str']
				data['predictions'] = pred
				preds.append(data)
		return render_template('tests.html', files = filelists, predictions=preds)

		
# if __name__ == '__main__':
#     app.run()