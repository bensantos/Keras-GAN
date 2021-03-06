from prep_samples import load_real_samples
# load image data
A_data, B_data = load_real_samples(r'D:\Kaspar\monet_dataset/monet.npz')
print('Loaded', A_data[0].shape, B_data[1].shape)

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model(r'D:\Kaspar\monet_dataset\models\g_model_BtoA_047720.h5', cust)
model_BtoA = load_model('g_model_BtoA_089025.h5', cust)

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()

# plot B->A->B
B_real = select_sample(B_data, 1)
print("check")
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)