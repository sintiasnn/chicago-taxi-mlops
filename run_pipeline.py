from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from taxi_pipeline.pipeline_def import create_pipeline


def main():
    pipeline = create_pipeline()
    BeamDagRunner().run(pipeline)


if __name__ == "__main__":
    main()
