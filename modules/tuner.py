"""Tuner module for TFX pipeline — performs hyperparameter tuning."""

from collections import namedtuple

import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.model import (
    _build_keras_model,
    _input_fn,
)

TUNER_TRIALS = 2
TunerFnResult = namedtuple("TunerFnResult", ["tuner", "fit_kwargs"])


def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=1, batch_size=64
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, num_epochs=1, batch_size=64
    )

    class_weight = {0: 0.73, 1: 1.58}

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: _build_keras_model(hp, tf_transform_output=tf_transform_output),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=TUNER_TRIALS,
        directory=fn_args.working_dir,
        project_name="taxi_tip_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "epochs": 10,
            "steps_per_epoch": fn_args.train_steps // 3,
            "validation_steps": fn_args.eval_steps // 3,
            "class_weight": class_weight,
        },
    )
