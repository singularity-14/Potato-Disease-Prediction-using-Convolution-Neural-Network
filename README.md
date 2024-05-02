# Potato Disease Prediction using Convolutional Neural Network (CNN)

## Overview

This project implements a Convolutional Neural Network (CNN) to analyze images of potato crops and predict the presence of specific diseases. The model is trained on a dataset containing images of healthy and diseased potato plants.

## Dataset

- **Dataset**: Kaggle
- **Training Data**: The training dataset consists of images of potato plants, categorized into healthy and diseased classes.
- **Validation Data**: The validation dataset is used to evaluate the performance of the trained model.

## Model Architecture

The CNN model architecture comprises several convolutional layers followed by max-pooling layers for feature extraction. The flattened output is then passed through fully connected layers for classification.

### Layers:
1. **Convolutional Layers**: Three convolutional layers with ReLU activation functions.
2. **Max-Pooling Layers**: Three max-pooling layers to downsample feature maps.
3. **Flatten Layer**: Flattens the output from convolutional layers.
4. **Dense Layers**: Two dense layers with ReLU activation functions and dropout regularization.
5. **Output Layer**: Final dense layer with a sigmoid activation function for binary classification.

## Training

- **Normalization**: The image data is normalized by dividing pixel values by 255 to scale them between 0 and 1.
- **Loss Function**: Binary cross-entropy loss is used as the loss function.
- **Optimizer**: Adam optimizer is employed for model optimization.
- **Metrics**: Accuracy is used as the evaluation metric.

## Usage

1. **Dataset Preparation**:
   - Organize your dataset into training and test sets, containing images of healthy and diseased potato plants.
   
2. **Model Training**:
   - Run the provided Python script `potato_disease_prediction.py`.
   - Ensure correct paths to the training and test datasets are specified.
   - Adjust hyperparameters as needed (e.g., number of epochs, batch size).
   
3. **Evaluation**:
   - The script will output training and validation accuracy and loss plots, indicating the model's performance over epochs.

## Customization

- Experiment with different CNN architectures, such as varying the number of layers or filter sizes.
- Fine-tune hyperparameters like learning rate, dropout rate, and batch size to improve model performance.
- Augment the dataset with techniques like rotation, flipping, or adding noise to increase model robustness.

## File Structure

- `potato_disease_prediction.py`: Main Python script for model training and evaluation.
- `README.md`: This file providing an overview and instructions.
- `potato_early_late/`: Directory containing the dataset split into training and test sets.
  
## Dependencies

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV

## License

This project is licensed under the [MIT License](LICENSE).

## Author

- Rachit Patel
