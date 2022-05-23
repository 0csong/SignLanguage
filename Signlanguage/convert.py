# #일단 pb로 변환
# from tensorflow import keras
# model = keras.models.load_model('handtrain(400(91)).h5', compile=False)
#
# export_path = './pb'
# model.save(export_path, save_format="tf")

import tensorflow as tf

saved_model_dir = './pb'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('model/handtrain(400(91)).tflite', 'wb').write(tflite_model)