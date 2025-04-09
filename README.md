#   COVID-19 X-Ray Classifier

##   Table of Contents

1.  [Project Description](#project-description)
2.  [Technologies Used](#technologies-used)
3.  [Installing Requirements](#installing-requirements)
4.  [Features](#features)

##   1. Project Description

A Python-based deep learning application that classifies chest X-ray images as **CovidPositive** or **CovidNegative** using a pre-trained **ResNet18** model. The project includes functionality for training the model, validating its performance, and evaluating new X-ray images for inference. It demonstrates the application of transfer learning and image preprocessing techniques for binary classification tasks in the medical imaging domain.

##   2. Technologies Used

* Python 3.10+
* [PyTorch](https://pytorch.org/) (deep learning framework)
* [Torchvision](https://pytorch.org/vision/) (for pre-trained models and image datasets)
* [scikit-learn](https://scikit-learn.org/) (for calculating metrics like F1 score)
* [Pillow](https://python-pillow.org/) (for image loading and processing)
* [NumPy](https://numpy.org/) (for numerical operations)

##   3. Installing Requirements

To install all required dependencies, run:

```bash
pip install -r requirements.txt
