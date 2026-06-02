#!/bin/bash

echo "Setting up ML Pipeline Project..."

conda create -n chicago-taxi-mlops python=3.9 -y
conda activate chicago-taxi-mlops

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the pipeline:"
echo "  conda activate chicago-taxi-mlops"
echo "  python run_pipeline.py"
echo ""
echo "To serve the model (via TF Serving Docker):"
echo "  docker run -p 8501:8501 \\"
echo "    -v \$(pwd)/serving_model:/models/taxi-model \\"
echo "    -e MODEL_NAME=taxi-model \\"
echo "    tensorflow/serving"
