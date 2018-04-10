import tensorflow as tf

import images

class Summary:
    """
        Handle tensorflow summaries.
    """

    def __init__(self, options, session, summary_path):
        self._options = options
        self._session = session
        self._summary_writer = tf.summary.FileWriter(summary_path, session.graph)
        print("================ Created summary writer")
        self.summary_ops = []

    def flush(self):
        """

        """
        print("============== Summary flushed")
        self._summary_writer.flush()

    def add(self, summary_str, global_step=None):
        """

        """
        self._summary_writer.add_summary(summary_str, global_step)

    def get_summary_op(self, scalars):
        """

        """
        for key, value in scalars.items():
            self.summary_ops.append(tf.summary.scalar(key, value))

        return tf.summary.merge(self.summary_ops)

    def initialize_train_summary(self):
        self._train_predictions = tf.placeholder(tf.int64, name="train_predictions")
        self._train_labels = tf.placeholder(tf.int64, name="train_labels")

        predictions = self._train_predictions
        labels = self._train_labels

        accuracy, recall, precision, f1_score = self.get_prediction_metrics(labels, predictions)

        self._train_summary = [tf.summary.scalar("train accuracy", accuracy)]
        self._train_summary.append(tf.summary.scalar("train recall", recall))
        self._train_summary.append(tf.summary.scalar("train precision", precision))
        self._train_summary.append(tf.summary.scalar("train f1_score", f1_score))
        self._train_summary = tf.summary.merge(self._train_summary)


    def initialize_missclassification_summary(self):
        self._missclassification_rate = tf.placeholder(tf.float64, name='misclassification_rate')
        self._misclassification_summary = tf.summary.scalar('misclassification_rate', self._missclassification_rate)

    def add_to_training_summary(self, predictions, labels, global_step):
        train_predictions = predictions
        train_labels = labels

        feed_dict_train = {
            self._train_predictions: train_predictions,
            self._train_labels: train_labels
        }

        train_sum, step = self._session.run([self._train_summary, global_step],
                                            feed_dict=feed_dict_train)
        self._summary_writer.add_summary(train_sum, global_step=step)

    def add_to_pixel_missclassification_summary(self, num_errors, total, global_step):
        misclassification, step = self._session.run([self._misclassification_summary, global_step],
                                                    feed_dict={self._missclassification_rate: num_errors / total})
        self._summary_writer.add_summary(misclassification, global_step=step)

    def get_prediction_metrics(self, labels, predictions):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)[1]
        recall = tf.metrics.recall(labels=labels, predictions=predictions)[1]
        precision = tf.metrics.precision(labels=labels, predictions=predictions)[1]
        f1_score = 2 / (1 / recall + 1 / precision)

        return accuracy, recall, precision, f1_score
