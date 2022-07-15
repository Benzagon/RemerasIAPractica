import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_prueba = datos['train'], datos['test']

nombres_clases = metadatos.features['label'].names

#Normalizar datos (pasar de 0-255 a 0-1)
def normalizer(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255 #Aqui se convierten
    return imagenes, etiquetas

#Normalizar datos de entrenamiento y prueba con la funcion que hicimos
datos_entrenamiento = datos_entrenamiento.map(normalizer)
datos_prueba = datos_prueba.map(normalizer)

#Agregar a cache (usar memoria en lugar de disco)
datos_entrenamiento = datos_entrenamiento.cache()
datos_prueba = datos_prueba.cache()

#Mostrar una imagen de nuestros datos_prueba, por ahora solo la primera
for imagen, etiqueta in datos_entrenamiento.take(1):
    break
imagen = imagen.numpy().reshape((28,28))

import matplotlib.pyplot as plt
#Dibujar
plt.figure()
plt.imshow(imagen, cmap = plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show