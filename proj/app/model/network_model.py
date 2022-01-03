# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:10:20 2021

@author: segatto
"""
# import libraries
import json
import time
import os
import numpy as np
from tensorflow import keras
# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# scikit
from scikitplot.metrics import plot_confusion_matrix
# my modules
from data.data_reader import get_labels, get_features, one_hot_encode_emoDB_labels, get_maps


def get_callbacks(model_name):
    """Instantiate callbacks objects for logging tensorboard results and saving model weights."""
    # Setup callbacks
    tensorboard_callback = TensorBoard(log_dir=f'model/logs/{model_name}')
    model_checkpoint_callback = ModelCheckpoint(
        filepath="model/weights/weights.best.hdf5",
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True)

    return tensorboard_callback, model_checkpoint_callback


def network_model():
    """
    CNN model implemented in Dias Issa et al 2020 Speech emotion recognition
    with deep convolutional neural networks (Model A: model with 7 classes).
    """

    # Define architectural params
    n_classes = 7
    stride = 1
    kernel_size = 5

    # New model
    model = Sequential()
    # The first layer of our CNN receives 193 × 1 number arrays as input data
    # The initial layer iscomposed of 256 filters with the kernel size of 5 × 5 and stride 1.
    model.add(Conv1D(filters=256, kernel_size=kernel_size, strides=stride,
                     padding='same', input_shape=(193, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=stride,
                     padding='same'))
    model.add(Activation('relu'))

    model.add(Dropout(0.1))
    # model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=(8)))

    # model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=stride,
    #                  padding='same'))
    # model.add(Activation('relu'))

    # model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=stride,
    #                  padding='same'))
    # model.add(Activation('relu'))

    model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=stride,
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # model.add(Conv1D(filters=128, kernel_size=kernel_size, strides=stride,
    #                  padding='same'))
    # model.add(Activation('relu'))

    model.add(Flatten())
    # model.add(Dropout(0.2))

    # Edit according to target class no.
    model.add(Dense(n_classes))
    # model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.summary()

    return model


def presence_of_pretrained_network_weights(weights_path="model/weights"):
    """Check if weights are present in weights_path. Returns true if so"""
    if os.path.isdir(weights_path) and os.path.isfile("model/weights/weights.best.hdf5"):
        return True
    else:
        return False


def run_best_model_or_train_model(model, X_train, X_test, y_train, y_test,
                                  force_train=False, num_epoch=10, save_history=True, pretrained=False):
    """
    Function responsible of training the CNN. if pretrained = True the pretrained
    network whose weights I provided is loaded. If pretrained = False this function
    checks if some weights from previous train calls are present, and in that case
    they are loaded into the model. If force_train = False the model is simply
    evaluated to get accuracy, otherwise it is further trained. If no weights can
    be located a new model is trained.
    """
    if pretrained:
        print("Using a pretrained network ...\nLoading and displaying train/validation accuracy")
        # load weights
        model.load_weights(
            "report_files/pretrained/pretrained_weights.best.hdf5")
        # compile the model setting the categorical cross entropy loss and the above
        # defined optimizer
        opt = keras.optimizers.RMSprop(
            learning_rate=0.00001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        history = []

    # check if pretrained weights are present in the project folder
    elif presence_of_pretrained_network_weights("model/weights") and force_train == False:
        print("found weights ...\nLoading and improving training of this model")
        # load weights
        model.load_weights("model/weights/weights.best.hdf5")
        # compile the model setting the categorical cross entropy loss and the above
        # defined optimizer
        opt = keras.optimizers.RMSprop(
            learning_rate=0.00001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        # getting callbacks
        tensorboard_callback, model_checkpoint_callback = get_callbacks(
            model.name)

        # train the model (saving logs for tensorboard and weights for speeding up next runs)
        print(f"Training for {num_epoch} epochs ...")
        history = model.fit(
            x=X_train,
            y=y_train,
            epochs=num_epoch,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback, model_checkpoint_callback],
            use_multiprocessing=True,
            workers=8)

        if save_history:
            hist_json_file = f'model/logs/{model.name}/history.json'
            with open(hist_json_file, mode='w') as f:
                json.dump(history.history, f)

    else:  # need to train a new model
        if force_train:
            print('Forcing train of a new net ... Discarding previous best weights')
        else:
            print(
                'No pretrained model found... Training a new network using reference parameters')

        # set a unique name
        name = f'cnn-issa-ser-{int(time.time())}'
        model._name = name

        # define the RMSprop as in the ref paper
        opt = keras.optimizers.RMSprop(
            learning_rate=0.00001, decay=1e-6)

        # compile the model setting the categorical cross entropy loss and the above
        # defined optimizer
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        # getting callbacks
        tensorboard_callback, model_checkpoint_callback = get_callbacks(
            model.name)

        # train the model (saving logs for tensorboard and weights for speeding up next runs)
        print(f"Training for {num_epoch} epochs ...")
        history = model.fit(
            x=X_train,
            y=y_train,
            epochs=num_epoch,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[tensorboard_callback, model_checkpoint_callback],
            use_multiprocessing=True,
            workers=8)

        if save_history:
            hist_json_file = f'model/logs/{model.name}/history.json'
            with open(hist_json_file, mode='w') as f:
                json.dump(history.history, f)

    # evaluate model
    scores_train = model.evaluate(X_train, y_train, verbose=0)
    scores_test = model.evaluate(X_test, y_test, verbose=0)

    return scores_train, scores_test, history


def plot_emotion_confusion_matrix(model, X, y):
    """ Plot confusion matrix on the predictions from X given the true values y."""
    # get emotions
    from_label_to_emotion = get_maps(from_='label', to_='emotion')
    emo_labels = list(from_label_to_emotion.values())

    # predict and get probabilities over each class
    class_probabilities = model.predict(X)

    # get the argmax and display the most likely class
    predicted_class = np.argmax(class_probabilities, axis=1)
    predicted_emotion = [from_label_to_emotion[i] for i in predicted_class]

    # get the argmax of the test labels
    true_class = np.argmax(y, axis=1)
    true_emotion = [from_label_to_emotion[i] for i in true_class]

    # plot the confusion matrix
    plot_confusion_matrix(true_emotion, predicted_emotion, labels=emo_labels, normalize=True,
                          title='Test set predictions', figsize=(10, 10))


def predict_from_sample(sample_number, best_pretrained=False):
    """
    Provides a model prediction for the given sample number.
    If best_pretrained=True it uses the model I provided otherwise it checks for
    weights of previous train calls and uses that model for prediction
    """
    # check if the user wants to use the best pretrained model:
    if best_pretrained:
        print('Making predictions using the best pretrained model available ...')
        # if yes build a model with them
        model = network_model()
        # load weights
        model.load_weights(
            "report_files/pretrained/pretrained_weights.best.hdf5")

        # compile the model setting the categorical cross entropy loss and the above
        # defined optimizer
        opt = keras.optimizers.RMSprop(
            learning_rate=0.00001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

    # check if model weights are present from previous train calls
    elif presence_of_pretrained_network_weights(weights_path="model/weights"):
        print("model weights found ... Building and compiling the model")
        # if yes build a model with them
        model = network_model()
        # load weights
        model.load_weights("model/weights/weights.best.hdf5")

        # compile the model setting the categorical cross entropy loss and the above
        # defined optimizer
        opt = keras.optimizers.RMSprop(
            learning_rate=0.00001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
    else:
        msg = ['No pretrained network found ... Please visit /train endpoint and train a new network ...']
        print(msg)
        return msg
    # load the audio filenames and labels
    filenames, labels = get_labels(DATA_PATH='data/wav/')

    if sample_number < 0 or sample_number > len(filenames):
        msg = f'Sample must be in range(0,{len(filenames)})'
        return [msg]
    # subselect the sample
    file = filenames[sample_number]
    label = labels[sample_number]
    print(f'Making predictions on audio sample: {file}')

    # extract features
    X = get_features(file)

    # predict and get probabilities over each class
    class_probabilities = model.predict(X)

    # get the argmax and display the most likely class
    predicted_class = np.argmax(class_probabilities)

    # compare with label
    from_label_to_code = get_maps(from_='label', to_='code')
    from_code_to_emotion = get_maps(from_='code', to_='emotion')

    true_emotion = from_code_to_emotion[from_label_to_code[label]]
    predicted_emotion = from_code_to_emotion[from_label_to_code[predicted_class]]

    return [file, label, predicted_class, class_probabilities, true_emotion, predicted_emotion]


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())
    from data.data_reader import get_dataset, create_train_test_sets
    X, y = get_dataset(augment=True, n_modifications=3)
    X_train, X_test, y_train, y_test = create_train_test_sets(
        X, y, test_size=0.2, random_state=1, shuffle=True)

    model = network_model()
    scores_train, scores_test, history = run_best_model_or_train_model(model, X_train, X_test, y_train, y_test,
                                                                       force_train=False, num_epoch=10, save_history=True, pretrained=True)

    print(f"Model: {model.name}")
    print(f"Training {model.metrics_names[1]} : {scores_train[1]*100}%")
    print(f"Test {model.metrics_names[1]} : {scores_test[1]*100}%")

    n = 50
    print(f"Making prediction of audio sample number {n}")
    output = predict_from_sample(n, best_pretrained=False)
    if len(output) > 1:
        file, label, predicted_class, class_probabilities, true_emotion, predicted_emotion = output
        print(f"Predicting file: {file}")
        print(f"True emotion is: {true_emotion}")
        print(f"Predicted emotion is: {predicted_emotion}")
        print(f'Class probabilities: {class_probabilities}')
    else:
        print(output)
    print("Displaying confusion matrix")
    plot_emotion_confusion_matrix(model, X_test, y_test)
