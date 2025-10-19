# 🐶 Dog Breed Identifier
This project demonstrates how to build a robust **Transfer Learning** pipeline using **ResNet50** for classifying 120 dog breeds from the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). The final model achieves **~81% accuracy** on the validation set using only ~9,600 training images. The trained model is then deployed via a lightweight **Streamlit App** to perform custom image inference.

## 📌 Project Highlights

- **Model Architecture**: Pretrained ResNet50 + custom classifier head with `ReLU` and `Dropout`
- **Dataset**: 120 dog breeds (~12K training images)
- **Training Accuracy**: ~82.15%
- **Validation Accuracy**: ~80.51% at Epoch 29/30
- **Loss Curve**: Smooth convergence without overfitting
- **Inference Interface**: Web-powered `Streamlit` App
- **Prediction Output**: Predicted dog breed + Confidence score
- **Visual Feedback**: ✅ or ✳️☑️ label color based on confidence threshold

## 🧠 Model Training Summary

- **Base model**: `ResNet50` pretrained on ImageNet
- **Freezing**: All ResNet convolutional layers frozen
- **Classifier head**:
  ```python
  def setup_resnet50_for_transfer_learning(num_classes: int):
    model = models.resnet50(pretrained=True)
    
    # Freeze all parameters in the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer (classifier)
    # The 'fc' layer in ResNet50 is the final classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Getting Started

` pip install -r requirements.txt
`
# Project structure
.
├── Appv2.py                # Streamlit App
├── resnet50TL.ipynb        # Model training notebook
├── requirements.txt        # Dependencies
├── resnet50_best.pth       # Saved model
├── data/
│   └── train/              # 120 folders, each named e.g. 'n02110185-siberian_husky'
│   └── test/
├── custom_images/images/   # Upload folder for app inference

# Running the App
streamlit run Appv2.py

# Sample Inference

* Uploaded Image: golden.jpg
* Prediction: golden_retriever
* Confidence: ✅ A figure

# Future Improvements
* Fine-tune top layers of ResNet
* Evaluate ensemble or lightweight models (e.g., EfficientNet-B0)

# Dockerize for deployment

# Acknowledgements
* Stanford Dogs Dataset
* PyTorch
* Streamlit
* torchvision.models

# License
* This project is licensed under the MIT License — see `LICENSE` file for details.
