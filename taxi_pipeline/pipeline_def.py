from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.orchestration.pipeline import Pipeline

from taxi_pipeline.components import (
    create_example_gen,
    create_statistics_gen,
    create_schema_gen,
    create_example_validator,
    create_transform,
    create_tuner,
    create_trainer,
    create_resolver,
    create_evaluator,
    create_pusher,
)
from taxi_pipeline.configs import (
    PIPELINE_NAME,
    PIPELINE_ROOT,
    METADATA_PATH,
)


def create_pipeline():
    components = []

    example_gen = create_example_gen()
    components.append(example_gen)

    statistics_gen = create_statistics_gen(example_gen)
    components.append(statistics_gen)

    schema_gen = create_schema_gen(statistics_gen)
    components.append(schema_gen)

    example_validator = create_example_validator(
        statistics_gen, schema_gen
    )
    components.append(example_validator)

    transform = create_transform(example_gen, schema_gen)
    components.append(transform)

    tuner = create_tuner(transform, schema_gen)
    components.append(tuner)

    trainer = create_trainer(transform, schema_gen, tuner)
    components.append(trainer)

    resolver = create_resolver()
    components.append(resolver)

    evaluator = create_evaluator(example_gen, transform, trainer, resolver)
    components.append(evaluator)

    pusher = create_pusher(trainer, evaluator)
    components.append(pusher)

    return Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=components,
        enable_cache=False,
        metadata_connection_config=sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=[
            "--runner=DirectRunner",
            "--direct_running_mode=in_memory",
        ],
    )
