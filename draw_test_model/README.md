One of the first exercises in training neural networks is to create a feed-forward neural network for the classification of digits using MNIST data.
This Flask app provides an interface for "real life" testing of such models.  The user can draw digits, see the models' predictions, and also save the images for later testing of new models.

For the time being, it's mostly a tool I created to help myself interact with neural network models.  In the future it might evolve into an educational tool.



To use it:
1) Download the repository.
2) Use pip to install the python packages indicated in the requirements.txt file (the specific versions are probably not important).
3) Create the directories: models/ and static/savedtests/
4) Make or modify some of the models in models_pytorch.py or models_tensorflow.py
5) Train the models using train.py -- this can basically be used out of the box, you just need to modify the options in the top of the script
6) Modify the list of models you want to actually run in app.py at the top of the script
7) Run Flask (go to the directory, then flask run) and then open your browser to http://localhost:5000/


The app also allows for training and testing of classification of images of articles of clothing using Fashion MNIST data.
This should probably not be taken too seriously.
