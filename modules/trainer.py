"""Trainer module for TFX pipeline — trains and exports the model."""

import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.transform import (
    CATEGORICAL_FEATURES,
    ALL_NUMERIC,
    transformed_name,
)
from modules.model import (
    _build_keras_model,
    _input_fn,
)

RAW_FEATURE_NAMES = [
    "trip_miles", "fare", "trip_seconds", "trip_start_timestamp",
    "payment_type", "company", "trip_start_hour", "trip_start_day",
    "trip_start_month",
]


class TFExampleModel(tf.keras.Model):
    """Keras model that accepts serialized tf.Example strings.

    Parses the tf.Example, transforms via TFT, and runs inference.
    """
    def __init__(self, combined_model, raw_feature_spec, feature_names):
        super().__init__()
        self.combined_model = combined_model
        self.raw_feature_spec = {n: raw_feature_spec[n] for n in feature_names}
        self.feature_names = feature_names

    def call(self, inputs):
        parsed = tf.io.parse_example(inputs, self.raw_feature_spec)
        return self.combined_model(parsed)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Build a Keras model wrapping TFT + trained model, accepting tf.Example.

    Uses TFExampleModel subclass and Keras Functional API to create a proper
    model graph that avoids the @tf.function variable-capture issue.
    """
    tft_layer = tf_transform_output.transform_features_layer()
    raw_spec = tf_transform_output.raw_feature_spec()

    raw_inputs = {}
    for name in RAW_FEATURE_NAMES:
        dtype = raw_spec[name].dtype
        raw_inputs[name] = tf.keras.Input(shape=(1,), dtype=dtype, name=name)

    transformed = tft_layer(raw_inputs)
    output = model(transformed)

    combined = tf.keras.Model(inputs=raw_inputs, outputs=output)

    serialized_input = tf.keras.Input(shape=(), dtype=tf.string, name="examples")
    tf_example_model = TFExampleModel(combined, raw_spec, RAW_FEATURE_NAMES)
    tf_example_output = tf_example_model(serialized_input)
    serving_model = tf.keras.Model(inputs=serialized_input, outputs=tf_example_output)

    return serving_model


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, tf_transform_output, num_epochs=20, batch_size=64
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, tf_transform_output, num_epochs=1, batch_size=64
    )

    class_weight = {0: 0.73, 1: 1.58}

    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
        model = _build_keras_model(hp, tf_transform_output=tf_transform_output)
    else:
        model = _build_keras_model(tf_transform_output=tf_transform_output)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="max",
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir,
            update_freq="batch",
        ),
    ]

    model.fit(
        train_dataset,
        epochs=20,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    serving_model = _get_serve_tf_examples_fn(model, tf_transform_output)
    tf.saved_model.save(serving_model, fn_args.serving_model_dir)
