from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Conv2DTranspose, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from load_images import get_image_paths, create_dataset
import cv2
from tqdm import tqdm


class ContextEncoder():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.mask_height = 64
        self.mask_width = 64
        self.channels = 3
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=Adam(0.0002, 0.5),
            metrics = ['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=Adam(0.0002, 0.5))

    def build_generator(self):

        # Encoder
        img_input = Input(self.img_shape)
        conv1 = Conv2D(64, kernel_size=5, strides=2, padding="same")(img_input)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = BatchNormalization(momentum=0.8)(conv1)

        conv2 = Conv2D(128, kernel_size=5, strides=2, padding="same")(conv1)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = BatchNormalization(momentum=0.8)(conv2)

        conv3 = Conv2D(256, kernel_size=5, strides=2, padding="same")(conv2)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = BatchNormalization(momentum=0.8)(conv3)

        conv4 = Conv2D(512, kernel_size=5, strides=2, padding="same")(conv3)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = BatchNormalization(momentum=0.8)(conv4)

        conv5 = Conv2D(512, kernel_size=5, strides=2, padding="same")(conv4)
        conv5 = LeakyReLU(alpha=0.2)(conv5)
        conv5 = BatchNormalization(momentum=0.8)(conv5)

        conv6 = Conv2D(512, kernel_size=5, strides=2, padding="same")(conv5)
        conv6 = LeakyReLU(alpha=0.2)(conv6)
        conv6 = BatchNormalization(momentum=0.8)(conv6)

        conv7 = Conv2D(512, kernel_size=5, strides=2, padding="same")(conv6)
        conv7 = LeakyReLU(alpha=0.2)(conv7)
        conv7 = BatchNormalization(momentum=0.8)(conv7)

        
        conv8 = Conv2D(512, kernel_size=5, strides=2, padding="same")(conv7)
        conv8 = LeakyReLU(alpha=0.2)(conv8)


        # Decoder
        deconv1 = Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', activation='relu')(conv8)
        deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        deconv1 = BatchNormalization(momentum=0.8)(deconv1)
        deconv1 = concatenate([deconv1,conv7], axis = 3)

        deconv2 = Conv2DTranspose(1024, kernel_size=5, strides=2, padding='same', activation='relu')(deconv1)
        deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        deconv2 = BatchNormalization(momentum=0.8)(deconv2)
        deconv2 = concatenate([deconv2, conv6], axis = 3)

        deconv3 = Conv2DTranspose(1024, kernel_size=5, strides=2, padding='same', activation='relu')(deconv2)
        deconv3 = LeakyReLU(alpha=0.2)(deconv3)
        deconv3 = BatchNormalization(momentum=0.8)(deconv3)
        deconv3 = concatenate([deconv3, conv5], axis = 3)

        deconv4 = Conv2DTranspose(1024, kernel_size=5, strides=2, padding='same', activation='relu')(deconv3)
        deconv4 = LeakyReLU(alpha=0.2)(deconv4)
        deconv4 = BatchNormalization(momentum=0.8)(deconv4)
        deconv4 = concatenate([deconv4, conv4], axis = 3)

        deconv5 = Conv2DTranspose(1024, kernel_size=5, strides=2, padding='same', activation='relu')(deconv4)
        deconv5 = LeakyReLU(alpha=0.2)(deconv5)
        deconv5 = BatchNormalization(momentum=0.8)(deconv5)
        deconv5 = concatenate([deconv5, conv3], axis = 3)

        deconv6 = Conv2DTranspose(512, kernel_size=5, strides=2, padding='same', activation='relu')(deconv5)
        deconv6 = LeakyReLU(alpha=0.2)(deconv6)
        deconv6 = BatchNormalization(momentum=0.8)(deconv6)
        deconv6 = concatenate([deconv6, conv2], axis = 3)

        # deconv7 = Conv2DTranspose(256, kernel_size=5, strides=1, padding='same', activation='relu')(deconv6)
        # deconv7 = LeakyReLU(alpha=0.2)(deconv7)
        # deconv7 = BatchNormalization(momentum=0.8)(deconv7)
        # deconv7 = concatenate([deconv7, conv1], axis = 3)

        # deconv8 = Conv2DTranspose(128, kernel_size=5, strides=1, padding='same', activation='relu')(deconv7)
        # deconv8 = LeakyReLU(alpha=0.2)(deconv8)
        # deconv8 = BatchNormalization(momentum=0.8)(deconv8)


        out = Conv2D(3, kernel_size=5, strides=1, padding='same', activation='tanh')(deconv6) #deconv8 
        model = Model(img_input, out)
        model.summary()

        #masked_img = Input(shape=self.img_shape)
        #gen_missing = model(masked_img)

        #return Model(masked_img, gen_missing)
        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(512, kernel_size=4, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()

        img = Input(shape=self.missing_shape)
        validity = model(img)

        return Model(img, validity)

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)



    def train(self, train_data, epochs, batch_size=256, sample_interval=50, load = False):

        if load == True:
            self.generator = keras.models.load_model()
        #(X_train, y_train), (_, _) = cifar10.load_data()
        # Extract dogs and cats
        #X_cats = X_train[(y_train == 3).flatten()]
        #X_dogs = X_train[(y_train == 5).flatten()]
        #X_train = np.vstack((X_cats, X_dogs))

        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        #y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            valid = valid - (np.random.uniform(0,.07))
            fake = fake + (np.random.uniform(0,.07))
            tq = tqdm(range(int(len(train_data)/batch_size)), desc=f"Epoch: {epoch}")
            for ind in tq:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                imgs  = create_dataset(train_data, 256, 256, batch_size)
                imgs = imgs/127.5 - 1


                #idx = np.random.randint(0, X_train.shape[0], batch_size)
                #imgs = X_train[idx]

                masked_imgs, missing_parts, _ = self.mask_randomly(imgs)

                # Generate a batch of new images
                gen_missing = self.generator.predict(masked_imgs)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

                # Plot the progress
                tq.set_postfix_str("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
                #print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]), end='\r')
                
                # If at save interval => save generated image samples
                if ind % sample_interval == 0:
                    #idx = np.random.randint(0, X_train.shape[0], 6)
                    #imgs = X_train[idx]
                    self.sample_images(ind, epoch, imgs)
            self.save_model(epoch)

    def sample_images(self, ind, epoch, imgs):
        r, c = 3, 6

        masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        gen_missing = self.generator.predict(masked_imgs)

        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(cv2.cvtColor(imgs[i, :,:], cv2.COLOR_BGR2RGB))
            axs[0,i].axis('off')
            axs[1,i].imshow(cv2.cvtColor(masked_imgs[i, :,:], cv2.COLOR_BGR2RGB))
            axs[1,i].axis('off')
            filled_in = imgs[i].copy()
            filled_in[y1[i]:y2[i], x1[i]:x2[i], :] = gen_missing[i]
            axs[2,i].imshow(cv2.cvtColor(filled_in, cv2.COLOR_BGR2RGB))
            axs[2,i].axis('off')
        fig.savefig("images/%d_%d.png" % (epoch,ind))
        plt.close()

    def save_model(self, epoch):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, ("generator" + str(epoch)))
        save(self.discriminator, ("discriminator" + str(epoch)))
        save(self.generator, ("gan" + str(epoch)))


if __name__ == '__main__':
    paths = get_image_paths(r"/home/ben/gans_git/Keras-GAN/landscapes")
    context_encoder = ContextEncoder()
    context_encoder.train(paths, epochs= 100, batch_size=256, sample_interval=50, load = False)
    context_encoder.save_model()

