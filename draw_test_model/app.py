from flask import Flask, render_template, request,  url_for, flash, redirect
from werkzeug.exceptions import abort
import tensorflow as tf
import numpy as np
import random

import torch
from torch import nn

app = Flask(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



digitsModels = ["digits.conv2.nll.adam.20.pth", "digits.conv2.nll.adam.30.pth", "digits.conv2.nll.100.keras"]
fashionModels = ["fashion.dense2.keras", "fashion.conv2.keras"]


@app.route('/', methods=('GET', 'POST'))
	



def index():
	if request.method == 'POST':

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
				pixels[i][j] = darkness
				imageData += "0, 0, 0, " + str(darkness) + ","
		imageData = imageData[0:-1] 	# Get rid of the last comma
		
		# What are we trying to classify?
		if request.form['type'] == "digit":
			textkeys = ["0", "1", "2","3","4","5","6","7","8","9"]
			names = digitsModels
		elif request.form['type'] == "fashion":
			textkeys = ["T-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
			names = fashionModels
		else:
			return render_template('index.html', notice="Something went wrong and you somehow requested a model we don't have.")
		
		# Load models
		models = []
		preds = []
		for s in names:
			pred = {'name': s}
			ext = s.split(".")[-1]
			if ext == "keras":
				m = tf.keras.models.load_model(s)
				prediction = m(tf.convert_to_tensor([pixels])).numpy()[0]		# The extra bracket makes it batch size 1, and the model expects non-scaled and no channels
			elif ext == "pth":
				modeltype = s.split(".")[1]
				# if modeltype == "conv2":
				# 	m = Conv2()
				# elif modeltype == "dense2":
				# 	m = Dense2()
				# else:
				# 	return render_template('index.html', notice="PyTorch model name not recognized.")
				#m.load_state_dict(torch.load(s))
				m = torch.jit.load(s)
				m.eval()					# Set to evaluation mode
				prediction = torch.nn.Softmax()(m(torch.tensor([[pixels]],dtype=torch.float) / 255.)).detach().numpy()[0]	# The extra bracket adds the channel and batch size 1, and the model expects a rescaled version
			else:
				return render_template('index.html', notice="Unrecognized file extension.")

			# Find the most likely prediction and its probability
			maxIndex = 0
			pred['probs'] = {}
			for i in range(0, len(prediction)):
				if prediction[maxIndex] < prediction[i]:
					maxIndex = i
				pred['probs'][textkeys[i]] = round(prediction[i] * 100, 2)
			pred['prob'] = round(prediction[maxIndex] * 100, 2)

			# Find the corresponding user-readable string corresponding to the most likely prediction
			pred['str'] = textkeys[maxIndex]
			preds.append(pred)
		
		return render_template('index.html', notice="Results:", results=preds, image=imageData)
	else:
		# randomDigits = random.choices(tf.keras.datasets.mnist.load_data()[0], k=10)
		# randomDigitsData = []
		# for p in range(0, len(randomDigits)):
		# 		imageData = ""
		# 		for i in range(0, 28):
		# 			for j in range(0, 28):
		# 				darkness = (np.ones(15) @ p[15*i:15*(i+1),15*j:15*(j+1)] @ np.ones(15)) / 225.0 * 255.0			# 15 * 15 = 225, note comes negative
		# 				imageData += "0, 0, 0, " + str(round(darkness)) + ","
		# 		imageData = imageData[0:-1] 	# Get rid of the last comma
		# 		randomDigitsData.append(imageData)
		return render_template('index.html')


if __name__ == '__main__':
    app.run()
