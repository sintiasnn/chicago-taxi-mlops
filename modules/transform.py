"""Transform module for TFX pipeline — feature engineering and preprocessing."""

import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURES = [
    "trip_miles",
    "fare",
    "trip_seconds",
    "trip_start_timestamp",
]

ENGINEERED_NUMERIC_FEATURES = [
    "trip_speed",
    "fare_per_mile",
    "log_trip_miles",
    "log_fare",
    "log_trip_seconds",
]

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

CATEGORICAL_FEATURES = {
    "payment_type": 10,
    "company": 100,
    "trip_start_hour": 24,
    "trip_start_day": 7,
    "trip_start_month": 12,
}

LABEL_KEY = "tips"


def transformed_name(key):
    return key + "_xf"


def _fill_in_missing(x):
    if not isinstance(x, tf.SparseTensor):
        return x
    return tf.sparse.to_dense(x, default_value="" if x.dtype == tf.string else 0)


STRING_CATEGORICAL_FEATURES = {"payment_type", "company"}


def preprocessing_fn(inputs):
    outputs = {}

    trip_miles = tf.cast(_fill_in_missing(inputs["trip_miles"]), tf.float32)
    trip_miles = tf.reshape(trip_miles, [-1, 1])
    trip_seconds = tf.cast(_fill_in_missing(inputs["trip_seconds"]), tf.float32)
    trip_seconds = tf.reshape(trip_seconds, [-1, 1])
    fare = tf.cast(_fill_in_missing(inputs["fare"]), tf.float32)
    fare = tf.reshape(fare, [-1, 1])

    outputs[transformed_name("trip_miles")] = tft.scale_to_z_score(trip_miles)
    outputs[transformed_name("fare")] = tft.scale_to_z_score(fare)
    outputs[transformed_name("trip_seconds")] = tft.scale_to_z_score(trip_seconds)

    ts = tf.cast(_fill_in_missing(inputs["trip_start_timestamp"]), tf.float32)
    ts = tf.reshape(ts, [-1, 1])
    outputs[transformed_name("trip_start_timestamp")] = tft.scale_to_z_score(ts)

    trip_speed = trip_miles / (trip_seconds + 1e-6)
    trip_speed = tf.clip_by_value(trip_speed, 0.0, 100.0)
    outputs[transformed_name("trip_speed")] = tft.scale_to_z_score(trip_speed)

    fare_per_mile = fare / (trip_miles + 1e-6)
    fare_per_mile = tf.clip_by_value(fare_per_mile, 0.0, 100.0)
    outputs[transformed_name("fare_per_mile")] = tft.scale_to_z_score(fare_per_mile)

    outputs[transformed_name("log_trip_miles")] = tft.scale_to_z_score(
        tf.math.log1p(trip_miles))
    outputs[transformed_name("log_fare")] = tft.scale_to_z_score(
        tf.math.log1p(fare))
    outputs[transformed_name("log_trip_seconds")] = tft.scale_to_z_score(
        tf.math.log1p(trip_seconds))

    for feature in CATEGORICAL_FEATURES:
        x = tf.reshape(_fill_in_missing(inputs[feature]), [-1, 1])
        if feature in STRING_CATEGORICAL_FEATURES:
            outputs[transformed_name(feature)] = (
                tft.compute_and_apply_vocabulary(x, vocab_filename=transformed_name(feature))
            )
        else:
            outputs[transformed_name(feature)] = tf.cast(x, tf.int64)

    label = tf.reshape(_fill_in_missing(inputs[LABEL_KEY]), [-1, 1])
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.greater(label, tf.constant(0.0)), tf.int64
    )

    return outputs
