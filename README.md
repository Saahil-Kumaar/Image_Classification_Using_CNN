# Image Classification using CNN - 🐶Dogs vs Cats🐱

This project implements a **Convolutional Neural Network (CNN)** to classify images of dogs and cats using TensorFlow and Keras. It uses the "Dogs vs Cats" dataset from Kaggle and includes preprocessing, model building, training, evaluation, and prediction on custom images.

---

## 📁 Dataset

The dataset used is:
- [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

It consists of images of dogs and cats divided into training and testing directories.

---

## 📦 Requirements

To run this project, you need the following dependencies:

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Kaggle CLI

Install the required packages:

```bash
pip install tensorflow keras opencv-python matplotlib kaggle
````

---

## 🚀 Setup Instructions

### 🔑 Step 1: Configure Kaggle API

1. Download your `kaggle.json` from your Kaggle account.
2. Move it to the correct location and set permissions:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 📥 Step 2: Download and Extract Dataset

```bash
!kaggle datasets download -d salader/dogs-vs-cats
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
```

---

## 🧠 Model Architecture

A CNN model was built with the following layers:

* 3 Convolutional layers with ReLU activation
* Batch Normalization
* MaxPooling after each conv block
* Fully Connected Dense layers
* Dropout for regularization
* Sigmoid output layer for binary classification

### 🔧 Model Summary

* Input size: `256x256x3`
* Loss: `Binary Crossentropy`
* Optimizer: `Adam`
* Output: `1` (Sigmoid for binary classification)

---

## 🏋️ Training the Model

```python
history = model.fit(train_ds, epochs=10, validation_data=test_ds)
```

Training and validation accuracy/loss are visualized using `matplotlib`.

---

## 📊 Results

Accuracy and loss graphs are plotted for both training and validation sets to monitor overfitting and convergence.

---

## 🖼️ Testing on New Images

You can test your own images by using OpenCV:

```python
test_img = cv2.imread('/content/dog.jpg')
test_img = cv2.resize(test_img, (256, 256))
test_img = test_img / 255.0
test_img = test_img.reshape((1, 256, 256, 3))

prediction = model.predict(test_img)
print("Predicted class:", "Dog" if prediction >= 0.5 else "Cat")
```

---

## 📈 Sample Output

```
Predicted class: Dog (with prediction value : [[0.9123]])
Predicted class: Cat (with prediction value : [[0.1021]])
```

---

## 📂 Directory Structure

```
├── kaggle.json
├── dogs-vs-cats.zip
├── /content
│   ├── /train
│   │   ├── dog.1.jpg
│   │   └── cat.1.jpg
│   ├── /test
│   │   ├── dog.2.jpg
│   │   └── cat.2.jpg
```

---

## ✅ Future Improvements

* Data augmentation to increase generalization
* Hyperparameter tuning
* Use of transfer learning (e.g., VGG16, ResNet)
* Convert to a web app using Streamlit or Flask

---

## 📌 License

This project is for educational purposes. Dataset used is under Kaggle's license.

---

## 🙌 Acknowledgments

* [Kaggle Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
* TensorFlow & Keras community

---

## 💡 Author

**Sahil Kumar**
[GitHub](https://github.com/Saahil-Kumaar)

```

---

Let me know if you'd like to add model accuracy screenshots, Streamlit UI, or link this to a Colab notebook!
```
