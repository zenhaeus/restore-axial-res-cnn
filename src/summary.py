import tensorflow as tf

import images

class Summary:
    """
        Handle tensorflow summaries.
    """

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self.summary_ops = []
        self._summary_writer = None

    def create_writer(self, summary_path):
        self._summary_writer = tf.summary.FileWriter(summary_path, graph=self._session.graph)

    def flush(self):
        """

        """
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
        # TODO: rework this for validation statistics
        self._train_predictions = tf.placeholder(tf.float32, name="train_predictions")
        self._train_labels = tf.placeholder(tf.float32, name="train_labels")

        predictions = self._train_predictions
        labels = self._train_labels

        accuracy, recall, precision, f1_score = self.get_prediction_metrics(labels, predictions)

        self._train_summary = [tf.summary.scalar("train_accuracy", accuracy)]
        self._train_summary.append(tf.summary.scalar("train_recall", recall))
        self._train_summary.append(tf.summary.scalar("train_precision", precision))
        self._train_summary.append(tf.summary.scalar("train_f1_score", f1_score))
        self._train_summary = tf.summary.merge(self._train_summary)

    def initialize_eval_summary(self):
        opts = self._options
        num_eval_images = 2
        self._groundtruth = tf.placeholder(tf.float32, name="groundtruth")
        self._downsampled = tf.placeholder(tf.float32, name="downsampled")
        self._predictions = tf.placeholder(tf.float32, name="prediction")
        self._snr_improvement = tf.placeholder(tf.float32, name="snr_improvement")

        self._eval_summary = [
            tf.summary.image('Groundtruth', self._groundtruth, max_outputs=num_eval_images),
            tf.summary.image('Downsampled', self._downsampled, max_outputs=num_eval_images),
            tf.summary.image('Prediction', self._predictions, max_outputs=num_eval_images),
            tf.summary.scalar('SNR Improvement', self._snr_improvement)
        ]

        self._eval_summary = tf.summary.merge(self._eval_summary)


    def initialize_snr_summary(self):
        self._snr = tf.placeholder(tf.float64, name='snr')
        self._snr_summary = tf.summary.scalar('SNR', self._snr)


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

    def add_to_snr_summary(self, snr, global_step):
        snr, step = self._session.run(
                [self._snr_summary, global_step],
                feed_dict={self._snr: snr}
        )
        self._summary_writer.add_summary(snr, global_step=step)

    def add_to_eval_summary(self, groundtruth, downsampled, predictions, global_step):
        opts = self._options

        snr_bicubic = images.psnr(groundtruth, downsampled)
        snr_prediction = images.psnr(groundtruth, predictions)
        print("SNR Bicubic: {}, SNR Prediction: {}".format(snr_bicubic, snr_prediction))
        snr_improvement = snr_prediction - snr_bicubic

        # Clip downsampled version for display
        downsampled[downsampled < 0] = 0
        downsampled[downsampled > 1] = 1

        feed_dict_eval = {
            self._groundtruth: groundtruth,
            self._downsampled: downsampled,
            self._predictions: predictions,
            self._snr_improvement: snr_improvement
        }

        image_sum, step = self._session.run([self._eval_summary, global_step],
                                            feed_dict=feed_dict_eval)
        self._summary_writer.add_summary(image_sum, global_step=step)

    def get_prediction_metrics(self, labels, predictions):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)[1]
        recall = tf.metrics.recall(labels=labels, predictions=predictions)[1]
        precision = tf.metrics.precision(labels=labels, predictions=predictions)[1]
        f1_score = 2 / (1 / recall + 1 / precision)

        return accuracy, recall, precision, f1_score
