# **COVID-19 X-Ray Classifier**

This project is a Python application that classifies X-ray images as **CovidPositive** or **CovidNegative** using a ResNet18 model. It includes functionality to train the model, validate it, and evaluate new images.

---

## **Requirements**

- Python 3.10 or later
- `torch` (for deep learning model training)
- `torchvision` (for dataset utilities and pretrained models)
- `scikit-learn` (for calculating F1 scores)
- `Pillow` (for image processing)
- `numpy` (for numerical operations)

To install all dependencies, run:

```bash
pip install -r requirements.txt
