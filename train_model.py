from data_loader import load_and_preprocess_data
from cnn_model import create_cnn_model

(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()

# Create model
cnn_model = create_cnn_model()

# Train the CNN model with validation on the test set
training_history = cnn_model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the trained model on the test set
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy}')

# Save the trained model
cnn_model.save('trained_digit_recognition_model.h5')
