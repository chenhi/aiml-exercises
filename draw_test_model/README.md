One of the first exercises in training neural networks is to create a feed-forward neural network for the classification of digits using MNIST data.
This Flask app provides an interface for "real life" testing of such models.  The user can draw digits, see the models' predictions, and also save the images for later testing of new models.

For the time being, it's mostly a tool I created to help myself interact with neural network models.  In the future it might evolve into an educational tool.



To use it, download the repository and create the directories: models/ and static/savedtests/

app.yaml, main.py are just for deploying to Google Cloud and can be ignored.



The app also allows for training and testing of classification of images of articles of clothing using Fashion MNIST data.
This should probably not be taken too seriously.
