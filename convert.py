from tensorflow import keras
import tensorflow as tf
model = keras.models.load_model('D:/Emotion3/bin/_mini_XCEPTION.31-0.67.hdf5', compile=False)#모델 수정하세요
#model = keras.models.load_model('emotion.h5', compile=False)
export_path = 'D:/Emotion3/bin'
model.summary();
# model.save(export_path,save_format="tf")


# saved_model_dir = 'D:/Emotion3/tlite'
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.experimental_new_converter = True
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT]
# saved_model_obj = tf.saved_model.load(export_dir=saved_model_dir)
# print(saved_model_obj.signatures.keys())
# tflite_quant_model = converter.convert()
# open('D:/Emotion3/tlite/converted_model3.tflite', 'wb').write(tflite_quant_model)
