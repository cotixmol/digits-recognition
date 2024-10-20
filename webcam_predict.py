import cv2
import numpy as np
import tensorflow as tf

def preprocess_webcam_image(image):
    resize_dim = (28, 28)
    gray_conversion_code = cv2.COLOR_BGR2GRAY
    interpolation_method = cv2.INTER_NEAREST
    
    gray_image = cv2.cvtColor(cv2.resize(image, resize_dim), gray_conversion_code)
    inverted_image = cv2.bitwise_not(gray_image)
    resized_image = cv2.resize(inverted_image, resize_dim, interpolation=interpolation_method)

    return resized_image, resized_image.reshape(1, 28, 28, 1)

def predict_from_webcam(model):
    image_from_webcam = cv2.VideoCapture(0) 
    
    if not image_from_webcam.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = image_from_webcam.read() 
        if not ret:
            print("Failed to grab frame")
            break
        
        display_image, cnn_input = preprocess_webcam_image(frame)
        predicted_digit = np.argmax(model.predict(cnn_input))
        
        # Resize the image to 1000x1000 first
        output_image_size = (700, 700)
        interpolation_method = cv2.INTER_NEAREST
        upscaled_image = cv2.resize(display_image, output_image_size, interpolation=interpolation_method)
        
        # Set the text properties for the prediction
        text_position = (50, 100)  
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 3
        text_color = (255, 255, 255)
        text_thickness = 3
        
        # Draw the text on the upscaled image
        cv2.putText(upscaled_image, f'Prediction: {predicted_digit}', text_position, text_font, text_size, text_color, text_thickness)
        
        # Display the upscaled image with the prediction
        cv2.imshow('Processed Image', upscaled_image)
        
        cv2.waitKey(10)
    
    image_from_webcam.release()
    cv2.destroyAllWindows()

# Load the trained model and run digit recognition from webcam
cnn_model = tf.keras.models.load_model('trained_digit_recognition_model.h5')
predict_from_webcam(cnn_model)
