FROM tensorflow/serving:latest

COPY ./serving_model /models/taxi-model
COPY ./config /model_config

ENV MODEL_NAME=taxi-model
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8501

RUN echo '#!/bin/bash\nset -e\nenv\ntensorflow_model_server \
--port=8500 \
--rest_api_port=${PORT} \
--model_name=${MODEL_NAME} \
--model_base_path=/models/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
