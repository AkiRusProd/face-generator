import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imageio


noise_vector_size = 100 #как в модели

image_size = 64 #как в модели
channels = 3

image_shape=(image_size,image_size,channels)

x_num = 5
y_num = 5
margin = 15


path='generated gifs/faces.gif'

generator = load_model('models/face_generator_model.h5')


def generate_images(generator, noise):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5 #[-1;1] => [0:1]

    images_array = np.full((x_num * (margin + image_size), y_num * (margin + image_size), channels), 255, dtype=np.uint8)

    gen_num = 0
    for i in range(y_num):
        for j in range(x_num):
            y=i*(margin + image_size)
            x=j*(margin + image_size)

            images_array[y:y+image_size,x:x+image_size] = generated_images[gen_num] * 255 
            gen_num+=1

    return cv2.cvtColor(images_array, cv2.COLOR_BGR2RGB)



def create_gif():
    steps = 10
    noise_vectors_interp = []
    imgs=[]

    noise_vector_1 = tf.random.normal([x_num * y_num, noise_vector_size])

    for step in range(steps):
         
        noise_vector_2 = tf.random.normal([x_num * y_num, noise_vector_size])

        noise_vectors_interp.append(np.linspace(noise_vector_1, noise_vector_2, 15))

        noise_vector_1 = noise_vector_2

        for vector in noise_vectors_interp[step]:
            imgs.append(generate_images(generator, vector))
    
    
    imageio.mimsave(path, imgs)




if __name__ == '__main__':
    create_gif()
