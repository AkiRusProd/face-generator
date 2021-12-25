import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model, load_model
import numpy as np
import cv2
import os
import time
from tqdm import tqdm

epochs = 20

image_size = 64
channels = 3

buffer_size = 60000
batch_size = 32
noise_vector_size = 100

x_num= 5
y_num= 5
margin = 15

image_path='UTKFace'
data_name  = f'training_data{image_size}x{image_size}.npy'
data_list=[]

if not os.path.isfile(data_name):
    for i in tqdm(os.listdir(image_path), desc='dirs',colour="green"):
        img=cv2.imread(image_path + '\\' + i)
        data = (cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC))
        data_list.append(data)

    data_list = np.asarray(data_list).astype(np.float32)
    data_list = data_list/127.5-1

    np.save(data_name, data_list)
else:
    data_list=np.load(data_name)
    

training_dataset = tf.data.Dataset.from_tensor_slices(data_list).shuffle(buffer_size).batch(batch_size)



def generator_model(noise_size, channels):
    
    model = Sequential()

    model.add(Dense(4*4*256,activation="relu",input_dim=noise_size))
    model.add(Reshape((4,4,256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    return model



def discriminator_model(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, 
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model



def generate_and_save_images(generator, epoch, noise):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5 #[-1;1] => [0:1]

    images_array=np.full((x_num*(margin + image_size), y_num*(margin+ image_size), channels), 255,dtype=np.uint8)
    
    gen_num = 0
    for i in range(y_num):
        for j in range(x_num):
            y=i*(margin + image_size)
            x=j*(margin + image_size)

            images_array[y:y+image_size,x:x+image_size] = generated_images[gen_num]*255 
            gen_num+=1

    cv2.imwrite(f'generated images/output_img{epoch}.jpg',  images_array)#cv2.cvtColor(images_array, cv2.COLOR_BGR2RGB)

generator = generator_model(noise_vector_size, channels)

image_shape=(image_size,image_size,channels)
discriminator = discriminator_model(image_shape)

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)



@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_vector_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(training_dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        noise = tf.random.normal([x_num * y_num, noise_vector_size])

        for image_batch in training_dataset:
            train_step(image_batch)

       
        generate_and_save_images(generator, epoch + 1, noise)

        print (f'Time for epoch {epoch + 1} is {time.time()-start} sec')

train(training_dataset, epochs)

generator.save(os.path.join('models/face_generator_model.h5'))
