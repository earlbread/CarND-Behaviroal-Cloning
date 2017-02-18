from keras.models import Sequential, save_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback

import helper


class SaveEpoch(Callback):
    """Extend class of keras Callback to save model every epoch.
    """
    def on_epoch_end(self, epoch, logs={}):
        save_model(self.model, '{}.h5'.format(epoch + 1))


def get_model():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(64, 64, 3)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('elu'))

    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('elu'))

    model.add(Dense(100))
    model.add(Activation('elu'))

    model.add(Dense(50))
    model.add(Activation('elu'))

    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    return model


if __name__ == '__main__':
    epochs = 20
    batch_size = 128

    model = get_model()
    model.compile('adam', 'mse')

    driving_data = helper.get_data_from_log()
    train, val = helper.train_validation_split(driving_data)

    train_gen = helper.generate_batch(train, batch_size)
    val_gen = helper.generate_batch(val, batch_size)

    samples_per_epoch = len(train)
    nb_val_samples = len(val)

    callbacks = [SaveEpoch()]

    history = model.fit_generator(train_gen,
                                  samples_per_epoch=samples_per_epoch,
                                  nb_epoch=epochs,
                                  callbacks=callbacks,
                                  validation_data=val_gen,
                                  nb_val_samples=nb_val_samples)

    save_model(model, 'model.h5')
