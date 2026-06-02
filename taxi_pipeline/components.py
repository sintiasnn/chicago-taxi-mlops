from tfx.components import (
    ImportExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Tuner,
    Evaluator,
    Pusher,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
import tensorflow_model_analysis as tfma
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from taxi_pipeline.configs import (
    DATA_ROOT,
    TRANSFORM_MODULE_FILE,
    TRAINER_MODULE_FILE,
    TUNER_MODULE_FILE,
    SERVING_MODEL_DIR,
    TRAIN_NUM_STEPS,
    EVAL_NUM_STEPS,
)


def create_example_gen():
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    return ImportExampleGen(
        input_base=DATA_ROOT,
        output_config=output,
    )


def create_statistics_gen(example_gen):
    return StatisticsGen(
        examples=example_gen.outputs["examples"],
    )


def create_schema_gen(statistics_gen):
    return SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
    )


def create_example_validator(statistics_gen, schema_gen):
    return ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )


def create_transform(example_gen, schema_gen):
    return Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=TRANSFORM_MODULE_FILE,
    )


def create_tuner(transform, schema_gen):
    return Tuner(
        module_file=TUNER_MODULE_FILE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"], num_steps=TRAIN_NUM_STEPS
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"], num_steps=EVAL_NUM_STEPS
        ),
    )


def create_trainer(transform, schema_gen, tuner=None):
    trainer_args = dict(
        module_file=TRAINER_MODULE_FILE,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"], num_steps=TRAIN_NUM_STEPS
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=["eval"], num_steps=EVAL_NUM_STEPS
        ),
    )
    if tuner:
        trainer_args["hyperparameters"] = tuner.outputs["best_hyperparameters"]
    return Trainer(**trainer_args)


def create_resolver():
    return Resolver(
        strategy_class=latest_blessed_model_strategy.LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")


def create_evaluator(example_gen, transform, trainer, resolver):
    return Evaluator(
        examples=transform.outputs["transformed_examples"],
        example_splits=["eval"],
        model=trainer.outputs["model"],
        baseline_model=resolver.outputs["model"],
        eval_config=tfma.EvalConfig(
            model_specs=[
                tfma.ModelSpec(
                    signature_name="transformed",
                    label_key="tips_xf",
                )
            ],
            metrics_specs=[
                tfma.MetricsSpec(
                    metrics=[
                        tfma.MetricConfig(
                            class_name="BinaryAccuracy",
                        ),
                        tfma.MetricConfig(
                            class_name="AUC",
                        ),
                    ]
                ),
            ],
            slicing_specs=[
                tfma.SlicingSpec(),
            ],
        ),
    )


def create_pusher(trainer, evaluator):
    return Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=SERVING_MODEL_DIR
            )
        ),
    )
