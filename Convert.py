from keras.models import load_model
import tensorflowjs as tfjs

model = load_model("final_epoch.h5")
tfjs.converters.save_keras_model(model, "model/model")