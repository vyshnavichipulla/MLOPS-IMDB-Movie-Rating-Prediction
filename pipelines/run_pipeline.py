import subprocess
import sys


def run_step(script):

    print("\n================================")
    print("Running:", script)
    print("================================")

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:

        print("\nPipeline failed at:", script)

        raise SystemExit(1)


print("================================")
print("IMDB MOVIE ML PIPELINE")
print("================================")


# Step 1: Preprocessing
run_step(
    "src/preprocess.py"
)


# Step 2: Model training
run_step(
    "src/train.py"
)


# Step 3: Model evaluation
run_step(
    "src/evaluate.py"
)


# Step 4: Save best model
run_step(
    "src/save_best_model.py"
)


print("\n================================")
print("PIPELINE COMPLETED SUCCESSFULLY")
print("================================")