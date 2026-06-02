import os
import base64
import requests
import random
import time
import logging
import tensorflow as tf

BASE_URL = os.environ.get(
    "TARGET_URL",
    "https://chicago-taxi-mlops-production.up.railway.app",
)
ENDPOINT = f"{BASE_URL}/v1/models/taxi-model:predict"

SAMPLE_DATA = [
    {"trip_start_month": 5, "trip_start_hour": 14, "trip_start_day": 3,
     "trip_start_timestamp": 1437076800, "trip_miles": 2.5, "trip_seconds": 600,
     "fare": 8.5, "payment_type": "Credit Card", "company": "Taxi Affiliation Services"},
    {"trip_start_month": 12, "trip_start_hour": 22, "trip_start_day": 5,
     "trip_start_timestamp": 1437076800, "trip_miles": 5.0, "trip_seconds": 1200,
     "fare": 15.0, "payment_type": "Cash", "company": "Chicago Elite Cab Corp."},
    {"trip_start_month": 3, "trip_start_hour": 8, "trip_start_day": 1,
     "trip_start_timestamp": 1437076800, "trip_miles": 1.2, "trip_seconds": 300,
     "fare": 5.5, "payment_type": "Credit Card", "company": "Chicago Elite Cab Corp."},
    {"trip_start_month": 7, "trip_start_hour": 18, "trip_start_day": 6,
     "trip_start_timestamp": 1437076800, "trip_miles": 3.8, "trip_seconds": 900,
     "fare": 12.0, "payment_type": "Cash", "company": "Taxi Affiliation Services"},
]


def _make_tf_example(inst):
    feature = {
        "trip_miles": tf.train.Feature(float_list=tf.train.FloatList(value=[inst["trip_miles"]])),
        "fare": tf.train.Feature(float_list=tf.train.FloatList(value=[inst["fare"]])),
        "trip_seconds": tf.train.Feature(int64_list=tf.train.Int64List(value=[inst["trip_seconds"]])),
        "trip_start_timestamp": tf.train.Feature(int64_list=tf.train.Int64List(value=[inst["trip_start_timestamp"]])),
        "payment_type": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inst["payment_type"].encode()])),
        "company": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inst["company"].encode()])),
        "trip_start_hour": tf.train.Feature(int64_list=tf.train.Int64List(value=[inst["trip_start_hour"]])),
        "trip_start_day": tf.train.Feature(int64_list=tf.train.Int64List(value=[inst["trip_start_day"]])),
        "trip_start_month": tf.train.Feature(int64_list=tf.train.Int64List(value=[inst["trip_start_month"]])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return base64.b64encode(example.SerializeToString()).decode("utf-8")


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

def main():
    logging.info("Traffic generator started — sending tf.Example requests every 5s")
    while True:
        instances = random.sample(SAMPLE_DATA, random.randint(1, 2))
        payload = {
            "instances": [
                {"examples": {"b64": _make_tf_example(inst)}}
                for inst in instances
            ]
        }
        try:
            resp = requests.post(ENDPOINT, json=payload, timeout=10)
            logging.info(
                "Status: %s | Latency: %.3fs | Payload: %s",
                resp.status_code, resp.elapsed.total_seconds(), payload,
            )
        except Exception as e:
            logging.error(f"Error: {e}")
        time.sleep(5)

if __name__ == "__main__":
    main()
