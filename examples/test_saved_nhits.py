from neuralforecast import NeuralForecast
import torch

print("CUDA available:", torch.cuda.is_available())

nf = NeuralForecast.load("models/neuralforecast/nhits_small")

# Force CPU inference
for model in nf.models:
    model.trainer_kwargs["accelerator"] = "cpu"
    model.trainer_kwargs["devices"] = 1

print("Model loaded successfully!")
print("Models:", [model.alias for model in nf.models])

predictions = nf.predict()

print(predictions.head(10))