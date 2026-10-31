import os
import pickle
import sys

model_path = "models/best_model.pkl"

print("===== LAB 11: DEPLOYMENT READINESS =====")

if not os.path.exists(model_path):
    print("ERROR: Best model not found!")
    sys.exit(1)

with open(model_path, "rb") as f:
    model = pickle.load(f)

os.makedirs("deployment", exist_ok=True)

with open("deployment/deployment_status.txt", "w") as f:
    f.write("Deployment Readiness: PASSED\n")
    f.write("Model: " + type(model).__name__ + "\n")
    f.write("Model Path: " + model_path + "\n")

print("Model loaded successfully!")
print("Deployment readiness completed!")
