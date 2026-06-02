import os

PIPELINE_NAME = "taxi_pipeline"

DATA_ROOT = "data/tfrecord"
TRANSFORM_MODULE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modules", "transform.py"
)
TRAINER_MODULE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modules", "trainer.py"
)
TUNER_MODULE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "modules", "tuner.py"
)

PIPELINE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "output", "pipeline_root"
)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
SERVING_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "serving_model"
)

TRAIN_NUM_STEPS = 500
EVAL_NUM_STEPS = 200
