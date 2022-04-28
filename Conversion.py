import tensorflow as tf
tflite_model = tf.keras.models.load_model("models/imageclassifier.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model) 
tfmodel = converter.convert() 
open ('generated.tflite' , "wb") .write(tfmodel)
