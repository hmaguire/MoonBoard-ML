# MoonBoard-ML
ML model for predicting the difficulty of MoonBoard climbs using FSDL
(fullstackdeeplearning.com) best practices.

**Planned project structure:**

* Code quality assurance with pre-commit hooks, GitHub Actions, and pytest
* Grading difficulty via a Transformer architecture
* Model implementation in PyTorch
* Model training and basic tracking and metrics via PyTorch Lightning and torchmetrics
* Experiment tracking, hyperparameter tuning, and model versioning with W&B
* Model packaging in TorchScript
* Predictor backend containerization via Docker and deployment as a microservice on AWS Lambda
* Basic load testing with Locust
* Pure Python frontend web application in Gradio
* Model monitoring with Gantry

**Current Progress:**
* Ingestion of raw MoonBoard climbs data into pyTorch Dataset under a pyTorch
Lightning DataModule (grade_predictor/data/mb2016.py).

**In Progress:**
* Train and test split of dataset and their dataloaders.
* Implement K-Fold Cross-Validation for training data.

**Upcoming:**
* Simple transformer model implementation
* Temporal position (start, middle, end) encoding
* 2D (x,y coords) relative position of holds embedding
* Investigate weighted loss function, oversampling and undersampling for imbalanced dataset

For an overview of the planned MoonBoard Grader application architecture, click the badge below to open
the FSDL interactive Jupyter notebook on Google Colab that demonstrates
their Text Recognizer application architecture which will be re-implemented.

<div align="center">
  <a href="http://fsdl.me/2022-overview"> <img src=https://colab.research.google.com/assets/colab-badge.svg width=240> </a>
</div> <br>