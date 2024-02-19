from flask import Flask, render_template, request,  url_for, flash, redirect
from werkzeug.exceptions import abort
import tensorflow as tf
import numpy as np

app = Flask(__name__)


digitsModels = ["digits.dense128", "digits.conv32x64"]
fashionModels = ["fashion.dense128", "digits.conv32x64"]

@app.route('/', methods=('GET', 'POST'))
def index():
	if request.method == 'POST':

		# Get pixels from the form in string form, then stick it in a (420, 420) array
		pixelstext = request.form['pixels']
		bigpixels = np.array([[0 for i in range(0, 420)] for j in range(0, 420)])
		for i in range(0, 420):
			for j in range(0, 420):
				bigpixels[i][j] = pixelstext[(j % 420) + (i * 420)]

		# Reduce size to (28, 28)
		pixels = np.array([[0. for i in range(0, 28)] for j in range(0, 28)])
		imageData = ""
		for i in range(0, 28):
			for j in range(0, 28):
				darkness = (np.ones(15) @ bigpixels[15*i:15*(i+1),15*j:15*(j+1)] @ np.ones(15)) / 225.0			# 15 * 15 = 225, note comes negative
				pixels[i][j] = darkness
				imageData += "0, 0, 0, " + str(round(darkness * 255.0)) + ","
		imageData = imageData[0:-1] 	# Get rid of the last comma
		
		# Load the right model and make the prediction
		models = []
		if request.form['type'] == "digit":
			for s in digitsModels:
				models.append(tf.keras.models.load_model(s + '.keras', safe_mode=False))
			textkeys = ["0", "1", "2","3","4","5","6","7","8","9"]
			names = digitsModels
		elif request.form['type'] == "fashion":
			for s in fashionModels:
				models.append(tf.keras.models.load_model(s + '.keras', safe_mode=False))
			textkeys = ["T-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
			names = fashionModels
		else:
			return render_template('index.html', notice="Something went wrong and you somehow requested a model we don't have.")
		
		preds = []
		for i in range(0, len(models)):
			pred = {}
			m = models[i]
			pred['name'] = names[i]
			prediction = m(tf.convert_to_tensor([pixels])).numpy()[0]

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
		return render_template('index.html')
