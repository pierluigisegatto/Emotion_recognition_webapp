# -*- coding: utf-8 -*-
# author: Segatto Pier Luigi
# email: pier.segatto@gmail.com

# Load libraries
from data.data_reader import get_dataset, create_train_test_sets, get_maps
from model.network_model import network_model, run_best_model_or_train_model, predict_from_sample
import os
from flask import Flask, jsonify, request

# initialize the web app object
app = Flask(__name__)

# and the api routes


@app.route("/")
def home():
    return "Visium project: visit /train and /predict endpoints to proceed"


@app.route("/train")
def train_model():
    # User modifiable parameters
    augment = True  # If the dataset has to be augmented
    n_modifications = 3  # how many times each audio will be augmented
    test_size = 0.2  # fraction of the dataset kept for test
    random_state = 1  # random state for reproducibility
    # set true to shuffle the dataset (according to the random state)
    shuffle = True
    force_train = False  # if true when issued a train request the previous model that has been queried will be improved with a new train
    num_epoch = 10  # num of epochs to train
    save_history = True  # set to true to log the loss and accuracy over each epoch
    pretrained = True  # set to true to use the pretrained model I provided

    # create the dataset and augment it if not already present or required
    X, y = get_dataset(augment=augment, n_modifications=n_modifications)
    X_train, X_test, y_train, y_test = create_train_test_sets(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    # log sizes
    train_shape = X_train.shape
    test_shape = X_test.shape

    # initialize the model
    model = network_model()
    # Train or get metrics for the requested model
    scores_train, scores_test, history = run_best_model_or_train_model(model, X_train, X_test, y_train, y_test,
                                                                       force_train=force_train, num_epoch=num_epoch, save_history=save_history,
                                                                       pretrained=pretrained)

    return jsonify({'Augmented dataset': augment,
                    'Number of times each audio has been augmented': n_modifications,
                    'Train size': train_shape,
                    'Test size': test_shape,
                    'Used the optimized best model': pretrained,
                    'Training has been forced': force_train,
                    'if training epochs just performed:': num_epoch,
                    f"Training {model.metrics_names[1]}": f"{scores_train[1]*100}%",
                    f"Test {model.metrics_names[1]}": f"{scores_test[1]*100}%"})


@app.route("/predict")
def predict_cnn():
    # User modifiable parameters
    best_pretrained = True

    # get the sample ID requested
    sample = request.args.get('ID')
    if sample is None:
        # display all available filenames and their ID
        files_and_indexes_available = {
            i: el for i, el in enumerate(os.listdir('data/wav'))}
        return jsonify(files_and_indexes_available)

    # get the prediction (or error message if input is wrong format)
    output = predict_from_sample(int(sample), best_pretrained=best_pretrained)

    # if len of output > 1 then no error msg has ben raised
    if len(output) > 1:
        # unpack the output
        file, label, predicted_class, class_probabilities, true_emotion, predicted_emotion = output
        # beautify the probabilities with the emotion labels
        emotions = list(get_maps(from_='label', to_='emotion').values())
        dict_class_probabilities = {i: j for i,
                                    j in zip(emotions, class_probabilities[0])}
        # create message
        message = {"Audio file": file,
                   "True emotion is": true_emotion,
                   "Predicted emotion is": predicted_emotion,
                   "Class probabilities": str(dict_class_probabilities),
                   'Used the optimized best model': best_pretrained}
        return jsonify(message)
    else:
        return jsonify({'Error:': output})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')  # remember to set debug to False
