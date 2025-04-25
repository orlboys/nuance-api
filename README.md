# Nuance-API

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


## Summary

- Nuance-API provides access to a DistilBERT-based model that can detect political bias in text.
- Nuance-API serves as the backend logic behind the greater Nuance project
	- [*Checkout orlboys/Nuance-Web for the frontend!*](https://github.com/orlboys/nuance-web)

*Developed as part of the HSC Software Engineering course*

## Structure

Nuance-API has two 'modules' tracked in the one repo:
- The API implementation, under app/
- The NLP model training algorithm, under training/

Both module's dependencies can be seen in their respective requirements_(module name).txt. We recommend setting up two separate python venv environments, as described below in *Setup*

## Setup

### App

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
    - Windows: `.\venv\Scripts\activate`
    - macOS/Linux: `source venv/bin/activate`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the app (using uvicorn): `uvicorn main:app`
5. Access [localhost:8000](localhost:8000) or [localhost:8000/docs](localhost:8000/docs) for debugging

### Training

1. Create a virtual environment: `python -m venv venv` (if you haven't already)
2. Activate the virtual environment:
    - Windows: `.\venv\Scripts\activate`
    - macOS/Linux: `source venv/bin/activate`
3. Install the required dependencies: `pip install -r requirements.txt` (from the project root)
4. Adjust `config.py` to declare how you want the model to be trained
5. Navigate to the training directory: `cd training`
6. Run the training script: `python train.py`
7. The new model's checkpoints will be stored in `training\trained_models\{model_name}\checkpoints`. TensorBoard logs are also mode in `training\runs`
