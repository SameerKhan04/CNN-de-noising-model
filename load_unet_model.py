import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Load the architecture and weights separately
# with open('weights/model_unet.json', 'r') as json_file:
#     model_json = json_file.read()
# model = model_from_json(model_json)
# model.load_weights('weights/model_unet.h5')

# Or load the complete model (uncomment the following line if using this method)
model = tf.keras.models.load_model('weights/model_best.keras')

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on a test dataset
# Ensure you have test_images and test_labels loaded or prepared
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


