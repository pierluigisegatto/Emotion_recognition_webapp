# Speech Emotion Recognition using CNN
This project implements the state-of-the-art **deep learning model** detailed in [Issa et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S1746809420300501?via%3Dihub):
*Speech emotion recognition with deep convolutional neural networks*.

**MODEL objective**: predict sentiments from audio files.
The model is trained on the [emo-DB](http://emodb.bilderbar.info/index-1280.html) Berlin dataset.

The proposed model uses a convolutional neural network (CNN) for the classification of emotions based on features extracted from a sound file.
The model includes one-dimensional convolutional layers combined with dropout, batch normalization, and activation layers.

On top of the model a **RESTFUL API** using Flask has been implemented. Particularly, two endpoints can be queried: */train* and */predict*.

The web service is deployed using a **docker container**. At runtime it exposes port 5000 to serve the Flask API and simultaneously port 8888 for the
Jupyter web server through which a python notebook named **Report_SER.ipynb** can be accessed to follow the data analysis and to observe
model results and responses from the queried API.

In order to run multiple services (Flask and Jupyter) from the same container I used [supervisor](http://supervisord.org/),
as suggested in the [docker documentation](https://docs.docker.com/config/containers/multi-service_container/).
Supervisor is a process control system that allows to control multiple processes defined in its *supervisord.conf* file.
When running the container supervisor creates the relative processes and can be run from a single docker entrypoint.


The following pipeline should be followed in order to build the docker image, run the container, access the notebook, and querying the API:

**NOTE**: I suppose docker is already installed in your local machine otherwise follow [these](https://docs.docker.com/get-docker/) instructions before proceeding.
- Build image:

    `docker build -t segatto .`

    This will create and prepare the entire project environment and tools (a debian based distro has been used as base image),
    downloads the raw dataset, install the required python packages and creates the docker image.
- Run the container:

    `docker run -d -p 5000:5000 -p 8888:8888 --name segatto_container segatto`

    This will run the docker entrypoint (uses supervisor) binding your local ports 5000 and 8888 to the same ports in the container.
    The container is run in detached mode (-d) to allow for querying the jupyter token needed to access the jupyter webservice.
- Access jupyter webserver:

    `docker exec -it segatto_container jupyter notebook list`

    Click on the link displayed to open jupyter. Navigate to *proj/app/* and open Report_SER.ipynb
- Querying the API and control model behavior:

    **NOTE**: All user modifiable parameters can be found in proj/app/app.py under the corresponding route functions.

    **TODO**: Trasform the data reader and network model modules into calsses (straight forward to do). Externalize all parameters to for easier modifications.

    - Train (GET method):

     `http://localhost:5000/train`

     The first time it is called the app augments the dataset (if the parameter augment = True) and creates the training and test sets. It will take some time to complete the process.
     Successive calls will use cached results.
     By default to seed up the tran process i provided an already pretrained model which is invoked when train is queryied (returns the model accuracy metrics).
     To train a new model set pretrained = False. Once the app finishes training (for num_epoch = 10 by default, but can be changed) the network state and metrics are returned.
     Successive calls to train will provide the same result if force_train=False otherwise will keep improving the training of the last model for a new set of epochs.

    - Predict (POST method):

     `http://localhost:5000/predict`

     Will return a list of filenames and the corresponding ID needed to be posted in the request in order to predict the sentiment of that specific audio file.

     `http://localhost:5000/predict?ID=52`

     Will return the model prediction of the audiofile whose id = 52. Set best_pretrained = False to use (if previously requested through a train query) the last trained model.
     Otherwise by default the API predicts using the pretrained and optimized model I provide.


