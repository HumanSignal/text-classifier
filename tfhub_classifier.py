import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import io
import json
import logging
import shutil

from pathlib import Path
from sklearn.cluster import KMeans

from htx.utils import encode_labels
from htx.base_model import SingleClassTextClassifier


logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(42)


class TFHubClassifier(SingleClassTextClassifier):

    def load(self, ser_train_output):
        train_output = json.loads(ser_train_output)
        export_dir = train_output.get('export_dir/latest')
        if export_dir:
            self._model = tf.contrib.predictor.from_saved_model(export_dir)

            # load idx2label
            idx2choice_file = train_output['idx2choice']
            if not os.path.exists(idx2choice_file):
                raise FileNotFoundError(f'Can\'t load label indices from {idx2choice_file}: file doesn\' exist.')
            with io.open(idx2choice_file) as f:
                self._idx2choice = json.load(f)
        else:
            raise FileNotFoundError(f'"export_dir/latest" field not found in resources {ser_train_output}')
        if not self._cluster:
            with io.open(train_output['cluster_file']) as f:
                self._cluster = json.load(f)

    def predict(self, tasks, **kwargs):
        """
        Return list of predictions given list of tasks
        :param tasks: list of tasks, where each task is {"input": ["text1", "text2", ...]}
        :param kwargs:
        :return: prediction result
        """
        input_texts = list(map(lambda item: item['input'][0], tasks))
        prediction = self._model({'text': input_texts})
        predict_proba = prediction['probabilities']
        predict_idx = np.argmax(predict_proba, axis=1)
        predict_scores = predict_proba[np.arange(len(predict_idx)), predict_idx].astype('float')
        predict_labels = [self._idx2choice[c] for c in predict_idx]
        results = self.make_results(tasks, predict_labels, predict_scores)
        for result, encoding in zip(results, prediction['encodings']):
            result['encoding'] = encoding
        return results

    @classmethod
    def _train_estimator_and_export_resources(cls, estimator, train_input_fn, model_dir, idx2choice, num_steps=1000):
        estimator.train(train_input_fn, steps=num_steps)
        logger.info(f'Training finished, output variables: {estimator.get_variable_names()}')

        feature_inputs = {'text': tf.placeholder(dtype=tf.string, shape=[None], name='text')}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_inputs)

        export_dir = os.path.join(model_dir, 'export')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        estimator.export_saved_model(export_dir, serving_input_receiver_fn)
        subdirs = [x for x in Path(export_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
        latest_subdir = str(sorted(subdirs)[-1])

        # clean temp checkpoints
        if os.path.exists(estimator.model_dir):
            logger.info(f'Clean up {estimator.model_dir}')
            shutil.rmtree(estimator.model_dir)

        idx2choice_file = os.path.join(model_dir, 'idx2choice.json')
        with io.open(idx2choice_file, mode='w') as fout:
            json.dump(idx2choice, fout, indent=4)

        resources = {
            'export_dir': export_dir,
            'export_dir/latest': latest_subdir,
            'idx2choice': idx2choice_file
        }

        return resources

    @classmethod
    def model_fn(cls, features, labels, mode, params):
        net_input = tf.feature_column.input_layer(features, params['feature_columns'])
        net = net_input
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        logits = tf.layers.dense(net, params['n_classes'], activation=None)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predicted_classes = tf.argmax(logits, 1)
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
                'encodings': net
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.003)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    @classmethod
    def cluster_kmeans(cls, ids, list_of_strings, k, tfhub_module_spec):
        logger.info(f'Clustering {len(list_of_strings)} items into {k} clusters')
        extractor = hub.Module(tfhub_module_spec)
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddings = sess.run(extractor(list_of_strings))
        kmeans = KMeans(n_clusters=k, n_jobs=-1, random_state=42)
        cluster_idx = list(map(int, kmeans.fit_predict(embeddings)))
        return dict(zip(ids, cluster_idx))

    @classmethod
    def fit_single_label(
        cls, input_data, output_model_dir,
        tfhub_module_spec='https://tfhub.dev/google/universal-sentence-encoder/2',
        num_steps=1000,
        layers=(200,),
        compute_clusters=True,
        **kwargs
    ):
        # collect data
        labeled_texts, labeled_ids, unlabeled_texts, unlabeled_ids, output_choices = [], [], [], [], []
        unique_choices = set()
        for item in input_data:
            text = item['input'][0]
            if item['output'] is None:
                unlabeled_texts.append(text)
                unlabeled_ids.append(item['id'])
            else:
                labeled_texts.append(text)
                labeled_ids.append(item['id'])
                output_choices.append(item['output'][0])
                unique_choices.add(item['output'][0])

        all_texts = unlabeled_texts + labeled_texts
        if compute_clusters:
            if len(unique_choices) == 0:
                num_clusters = min(100, len(all_texts))
            else:
                num_clusters = min(len(unique_choices) * 10, len(all_texts))
            cluster = cls.cluster_kmeans(unlabeled_ids + labeled_ids, all_texts, num_clusters, tfhub_module_spec)
        else:
            cluster = {}

        # create choices indexers
        idx2choice, output_choices_idx = encode_labels(output_choices)
        if len(idx2choice) < 2:
            raise ValueError(f'Unable to start training with less than two classes: {idx2choice}')
        else:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'text': np.array(labeled_texts)},
                y=np.array(output_choices_idx),
                shuffle=True,
                num_epochs=None
            )
            text_feature_column = hub.text_embedding_column(
                key='text',
                module_spec=tfhub_module_spec,
                trainable=False
            )

            estimator = tf.estimator.Estimator(
                model_fn=cls.model_fn,
                params={
                    'feature_columns': [text_feature_column],
                    'hidden_units': layers,
                    'n_classes': len(idx2choice)
                },
                config=tf.estimator.RunConfig(keep_checkpoint_max=1)
            )

            resources = cls._train_estimator_and_export_resources(
                estimator, train_input_fn, output_model_dir, idx2choice, num_steps
            )

        # save clusters
        cluster_file = os.path.join(output_model_dir, 'cluster.json')
        with io.open(cluster_file, mode='w') as fout:
            json.dump(cluster, fout, indent=2)
        resources['cluster_file'] = cluster_file

        logger.info(f'Model training finished, created new resources in {output_model_dir}: {resources}')
        return resources
