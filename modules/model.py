"""Shared model architecture and utilities for trainer and tuner."""

import tensorflow as tf

from modules.transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    ALL_NUMERIC,
    transformed_name,
)

EMBEDDING_DIM = 16


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64):
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def _get_vocab_size(tf_transform_output, feature):
    try:
        vocab = tf_transform_output.vocabulary_by_name(
            transformed_name(feature)
        )
        return len(vocab) + 1
    except (ValueError, KeyError, RuntimeError):
        return CATEGORICAL_FEATURES[feature] + 5


def _build_keras_model(hp=None, tf_transform_output=None):
    inputs = {}
    embedding_outputs = []

    for feature in ALL_NUMERIC:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,), name=transformed_name(feature)
        )

    for feature_name, vocab_max in CATEGORICAL_FEATURES.items():
        if tf_transform_output:
            vocab_size = _get_vocab_size(tf_transform_output, feature_name)
        else:
            vocab_size = vocab_max + 1
        cat_input = tf.keras.Input(
            shape=(1,),
            name=transformed_name(feature_name),
            dtype=tf.int64,
        )
        inputs[transformed_name(feature_name)] = cat_input
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            name=f"embedding_{feature_name}",
        )(cat_input)
        embedding_flat = tf.keras.layers.Flatten()(embedding)
        embedding_outputs.append(embedding_flat)

    numeric_inputs = [inputs[transformed_name(f)] for f in ALL_NUMERIC]
    numeric_concat = tf.keras.layers.concatenate(numeric_inputs)
    all_features = tf.keras.layers.concatenate(
        [numeric_concat] + embedding_outputs
    )

    units_1 = hp.Int("units_1", min_value=128, max_value=512, step=128) if hp else 256
    units_2 = hp.Int("units_2", min_value=64, max_value=256, step=64) if hp else 128
    units_3 = hp.Int("units_3", min_value=32, max_value=128, step=32) if hp else 64
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.4, step=0.1) if hp else 0.2
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]) if hp else 1e-3

    x = tf.keras.layers.Dense(units_1, activation="relu")(all_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units_2, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units_3, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
