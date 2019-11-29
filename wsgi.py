import os
import argparse

from tfhub_classifier import TFHubClassifier
from htx import app, init_model_server

init_model_server(
    create_model_func=TFHubClassifier,
    train_script=TFHubClassifier.fit_single_label,
    redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
    redis_host=os.environ.get('REDIS_HOST', 'localhost'),
    tfhub_module_spec=os.environ.get('tfhub_module_spec', 'https://tfhub.dev/google/universal-sentence-encoder/2')
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', default='9090')
    args = parser.parse_args()
    app.run(host='localhost', port=args.port, debug=True)
