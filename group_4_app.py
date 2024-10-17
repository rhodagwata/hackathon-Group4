import fastapi
import pydantic
import numpy as np
import joblib
import uvicorn
import logging
import random
import time
import hashlib

# Load the pre-trained model
svc_model = joblib.load('/home/sagemaker-user/group-4-agents/temp_repo/svc_model.pkl')

# Pydantic models
class IrisFeatures(pydantic.BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchIrisFeatures(pydantic.BaseModel):
    data: list[IrisFeatures]

# Prediction functions
async def predict(features: IrisFeatures):
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = svc_model.predict(data)
    species = ['setosa', 'versicolor', 'virginica'][prediction[0]]
    return species

async def predict_batch(batch: BatchIrisFeatures):
    data = np.array([[
        feature.sepal_length,
        feature.sepal_width,
        feature.petal_length,
        feature.petal_width
    ] for feature in batch.data])
    predictions = svc_model.predict(data)
    species = [[prediction] for prediction in predictions]
    return species

async def predict_random():
    sepal_length = random.uniform(4.3, 7.9)
    sepal_width = random.uniform(2.0, 4.4)
    petal_length = random.uniform(1.0, 6.9)
    petal_width = random.uniform(0.1, 2.5)
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = svc_model.predict(data)
    species = ['setosa', 'versicolor', 'virginica'][prediction[0]]
    return {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width,
        'species': species
    }

# Utility functions
async def health_check():
    return 'Healthy'

async def model_info():
    model_details = svc_model.get_params()
    return model_details

async def simulate_workload(seconds: int):
    start_time = time.time()
    while time.time() - start_time < seconds:
        await predict_random()

# FastAPI app
app = fastapi.FastAPI()

# API routes
@app.post('/predict')
async def predict_endpoint(features: IrisFeatures):
    try:
        species = await predict(features)
        return {'species': species}
    except Exception as e:
        logging.error(f'Error: {e}')
        return {'error': str(e)}

@app.post('/predict_batch')
async def predict_batch_endpoint(batch: BatchIrisFeatures):
    try:
        species = await predict_batch(batch)
        return {'species': species}
    except Exception as e:
        logging.error(f'Error: {e}')
        return {'error': str(e)}

@app.get('/predict_random')
async def predict_random_endpoint():
    try:
        result = await predict_random()
        return result
    except Exception as e:
        logging.error(f'Error: {e}')
        return {'error': str(e)}

@app.get('/health')
async def health_check_endpoint():
    return await health_check()

@app.get('/model_info')
async def model_info_endpoint():
    return await model_info()

@app.get('/simulate_workload')
async def simulate_workload_endpoint(seconds: int):
    await simulate_workload(seconds)
    return f'Simulated workload for {seconds} seconds'

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
    