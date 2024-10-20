import tensorflow as tf

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape for CNN input
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    
    return (train_images, train_labels), (test_images, test_labels)
