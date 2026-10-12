# iris-ml-cd-5

# Iris ML Continuous Delivery

A machine learning application that classifies Iris flower
species using Logistic Regression and Flask.

## Pipeline
1. Train the ML model
2. Check model quality
3. Run automated ML tests
4. Run Flask API tests
5. Prepare a release candidate
6. Upload the release artifact with GitHub Actions

## API
- GET / — Health check
- POST /predict — Predict Iris species

## Run locally
pip install -r requirements.txt
python iris_model.py
python quality_gate.py
python -m unittest test_iris_model.py -v
python -m unittest test_app.py -v
python app.py

## Release package
- app.py
- iris_model.pkl
- metrics.json
- requirements.txt
- DEPLOYMENT.txt
