"""Rebuild serving model with tf.Example input (TFX standard format).

Wraps the combined Keras model in a TFExampleModel that accepts serialized
tf.Example strings — the standard TF Serving format.
"""
import os
import sys
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_transform as tft

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RAW_FEATURE_NAMES = [
    "trip_miles", "fare", "trip_seconds", "trip_start_timestamp",
    "payment_type", "company", "trip_start_hour", "trip_start_day",
    "trip_start_month",
]


def _get_vocab_size(tf_transform_output, feature):
    from modules.transform import CATEGORICAL_FEATURES, transformed_name
    try:
        vocab = tf_transform_output.vocabulary_by_name(transformed_name(feature))
        return len(vocab) + 1
    except (ValueError, KeyError, RuntimeError):
        return CATEGORICAL_FEATURES[feature] + 5


def _build_trained_model(tf_transform_output):
    from modules.transform import CATEGORICAL_FEATURES, ALL_NUMERIC, transformed_name

    inputs = {}
    embedding_outputs = []

    for feature in ALL_NUMERIC:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature)
        )

    for feature_name in CATEGORICAL_FEATURES:
        vocab_size = _get_vocab_size(tf_transform_output, feature_name)
        cat_input = tf.keras.Input(
            shape=(1,), name=transformed_name(feature_name), dtype=tf.int64
        )
        inputs[transformed_name(feature_name)] = cat_input
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=16, name=f"embedding_{feature_name}",
        )(cat_input)
        embedding_outputs.append(tf.keras.layers.Flatten()(embedding))

    numeric_inputs = [inputs[transformed_name(f)] for f in ALL_NUMERIC]
    numeric_concat = tf.keras.layers.concatenate(numeric_inputs)
    all_features = tf.keras.layers.concatenate([numeric_concat] + embedding_outputs)

    x = tf.keras.layers.Dense(384, activation="relu")(all_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def _load_weights_by_position(model, ckpt_path):
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    ckpt_all = ckpt_reader.get_variable_to_shape_map()

    def sort_key(name):
        parts = name.split("/")
        op_num = int(parts[1])
        var_type = parts[2].lstrip("_")
        priority = {
            "kernel": 0, "embeddings": 0, "bias": 1, "gamma": 2,
            "beta": 3, "moving_mean": 4, "moving_variance": 5,
        }
        return (op_num, priority.get(var_type, 99))

    ckpt_vars = sorted(
        [(k, v) for k, v in ckpt_all.items()
         if not k.startswith("optimizer") and k != "_CHECKPOINTABLE_OBJECT_GRAPH"],
        key=lambda x: sort_key(x[0]),
    )

    weights = []
    for (ckpt_name, ckpt_shape), w in zip(ckpt_vars, model.weights):
        w_shape = list(w.shape)
        if ckpt_shape != w_shape:
            raise ValueError(
                f"Shape mismatch: {ckpt_name} {ckpt_shape} vs {w.name} {w_shape}"
            )
        weights.append(ckpt_reader.get_tensor(ckpt_name))

    model.set_weights(weights)
    logging.info("Loaded %d weights from checkpoint", len(weights))


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


def build_serving_model(transform_path, trainer_model_path, output_path):
    logging.info("Loading TFTransformOutput from %s", transform_path)
    tf_transform_output = tft.TFTransformOutput(transform_path)

    logging.info("Building trained model architecture")
    trained_model = _build_trained_model(tf_transform_output)

    ckpt_path = os.path.join(trainer_model_path, "variables/variables")
    logging.info("Loading weights from %s", ckpt_path)
    _load_weights_by_position(trained_model, ckpt_path)

    logging.info("Creating TFT transform layer")
    tft_layer = tf_transform_output.transform_features_layer()

    raw_spec = tf_transform_output.raw_feature_spec()

    raw_inputs = {}
    for name in RAW_FEATURE_NAMES:
        dtype = raw_spec[name].dtype
        raw_inputs[name] = tf.keras.Input(shape=(1,), dtype=dtype, name=name)

    transformed = tft_layer(raw_inputs)
    output = trained_model(transformed)

    combined_model = tf.keras.Model(inputs=raw_inputs, outputs=output)

    logging.info("Wrapping in TFExampleModel (tf.Example input)")
    serialized_input = tf.keras.Input(shape=(), dtype=tf.string, name="examples")
    tf_example_model = TFExampleModel(combined_model, raw_spec, RAW_FEATURE_NAMES)
    tf_example_output = tf_example_model(serialized_input)
    serving_model = tf.keras.Model(inputs=serialized_input, outputs=tf_example_output)

    version_path = os.path.join(output_path, "1")
    logging.info("Saving serving model to %s", version_path)
    tf.saved_model.save(serving_model, version_path)

    loaded = tf.saved_model.load(version_path)
    sig = loaded.signatures["serving_default"]
    inputs_info = dict(sig.structured_input_signature[1])
    logging.info("Saved model inputs: %s", list(inputs_info.keys()))
    logging.info("Total function inputs: %d", len(sig.inputs))

    feature = {
        "trip_miles": tf.train.Feature(float_list=tf.train.FloatList(value=[2.5])),
        "fare": tf.train.Feature(float_list=tf.train.FloatList(value=[8.5])),
        "trip_seconds": tf.train.Feature(int64_list=tf.train.Int64List(value=[600])),
        "trip_start_timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[1437076800])),
        "payment_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"Credit Card"])),
        "company": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"Taxi Affiliation Services"])),
        "trip_start_hour": tf.train.Feature(int64_list=tf.train.Int64List(value=[14])),
        "trip_start_day": tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
        "trip_start_month": tf.train.Feature(int64_list=tf.train.Int64List(value=[5])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    result = sig(tf.constant([serialized]))
    logging.info("Sanity check result: %s", result)

    return serving_model


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))

    pipeline_root = os.path.join(BASE, "output", "pipeline_root")

    latest_transform = sorted(
        d for d in os.listdir(os.path.join(pipeline_root, "Transform/transform_graph"))
        if d.isdigit()
    )[-1]

    latest_trainer_model = sorted(
        d for d in os.listdir(os.path.join(pipeline_root, "Trainer/model"))
        if d.isdigit()
    )[-1]

    transform_path = os.path.join(
        pipeline_root, "Transform/transform_graph", latest_transform
    )
    trainer_model_path = os.path.join(
        pipeline_root, "Trainer/model", latest_trainer_model, "Format-Serving"
    )
    output_path = os.path.join(BASE, "serving_model")

    logging.info("Latest transform: %s", latest_transform)
    logging.info("Latest trainer model: %s", latest_trainer_model)

    build_serving_model(transform_path, trainer_model_path, output_path)
