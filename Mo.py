import tensorflow as tf
import tensorflow_hub as hub


def predict_image_class(img_pa):
    model = tf.keras.models.load_model(("model.h5"), custom_objects={
                                       'KerasLayer': hub.KerasLayer})
    img = tf.keras.preprocessing.image.load_img(img_pa, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)  # Create a batch
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    predictions = model.predict(img)
    score = predictions.squeeze()
    if score < 0.29:
        print(f"This image is {100 * (1 - score):.2f}% benign.")
    elif score > 0.30 and score < 0.7:  
        print("Got nothing")
    elif score>0.71:
        print(f"This image is {100 * score:.2f}% malignant.")
