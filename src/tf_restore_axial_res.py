"""

"""
import os

import tensorflow as tf
import numpy as np

import unet
import images

from summary import Summary
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.app.flags.DEFINE_integer('gpu', 0, "GPU to run the model on")
tf.app.flags.DEFINE_integer('batch_size', 25, "Batch size of training instances")
# Valid sizes: [4, 36, 68, 100, 132, 164, 196, 228, 260, 292, 324, 356, 388, 420, 452, 484]
# TODO: test valid sizes
tf.app.flags.DEFINE_integer('patch_size', 100, "Size of the prediction image")
tf.app.flags.DEFINE_integer('stride', 16, "Sliding delta for patches")
tf.app.flags.DEFINE_integer('seed', 2018, "Random seed for reproducibility")
tf.app.flags.DEFINE_integer('root_size', 16, "Number of filters of the first U-Net layer")
tf.app.flags.DEFINE_integer('num_epoch', 5, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('num_layers', 3, "Number of layers of the U-Net")
tf.app.flags.DEFINE_integer('train_score_every', 1000, "Compute training score after the given number of iterations")

tf.app.flags.DEFINE_float('learning_rate', 0.01, "Initial learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('dropout', 0.8, "Probability to keep an input")

tf.app.flags.DEFINE_string('logdir', os.path.abspath("./logdir"), "Directory where to write logfiles")
tf.app.flags.DEFINE_string('save_path', os.path.abspath("./runs"),
                           "Directory where to write checkpoints, overlays and submissions")

FLAGS = tf.app.flags.FLAGS

class Options(object):
    """
        Options used by model
    """
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.gpu = FLAGS.gpu
        self.learning_rate = FLAGS.learning_rate
        self.logdir = FLAGS.logdir
        self.momentum = FLAGS.momentum
        self.patch_size = FLAGS.patch_size
        self.save_path = FLAGS.save_path
        self.seed = FLAGS.seed
        self.stride = FLAGS.stride
        self.dropout = FLAGS.dropout
        self.root_size = FLAGS.root_size
        self.num_epoch = FLAGS.num_epoch
        self.num_layers = FLAGS.num_layers
        self.train_score_every = FLAGS.train_score_every

class ConvolutionalModel:
    def __init__(self, options, session):
        self._options = options
        self._session = session

        np.random.seed(options.seed)
        tf.set_random_seed(options.seed)
        self.input_size = unet.input_size_needed(options.patch_size, options.num_layers)

        self.experiment_name = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
        experiment_path = os.path.abspath(os.path.join(options.save_path, self.experiment_name))
        summary_path = os.path.join(options.logdir, self.experiment_name)

        self._summary = Summary(options, session, summary_path)

        self.build_graph()

    def cross_entropy_loss(self, downsampled_patches, pred_logits):
        """

        """
        # TODO: figure out loss function
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred_logits,
            labels=downsampled_patches
        )
        loss = tf.reduce_mean(cross_entropy)

        return loss

    def optimize(self, loss):

        learning_rate = tf.train.exponential_decay(
                self._options.learning_rate,
                self._global_step,
                100,
                0.95,
                staircase=True
        )

        optimizer = tf.train.MomentumOptimizer(learning_rate, self._options.momentum)
        train = optimizer.minimize(loss, global_step=self._global_step)

        return train, learning_rate

    def build_graph(self):
        """

        """
        opts = self._options

        global_step = tf.Variable(0, name="global_step")
        self._global_step = global_step

        # data placeholders
        patches_node = tf.placeholder(
                tf.float32,
                shape=(
                    self._options.batch_size,
                    self.input_size,
                    self.input_size
                    # TODO: maybe multichannel for RGB data
                    #NUM_CHANNELS
                ),
                name="patches"
        )

        downsampled_patches_node = tf.placeholder(
                tf.int64,
                shape=(
                    self._options.batch_size,
                    self._options.patch_size,
                    self._options.patch_size
                ),
                name="downsampled_patches"
        )

        dropout_keep = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep")
        self._dropout_keep = dropout_keep

        predict_logits = unet.forward(patches_node, root_size=opts.root_size, num_layers=opts.num_layers,
                                      dropout_keep=dropout_keep)

        predictions = tf.nn.softmax(predict_logits)
        # drop channel axis
        predictions = predictions[:, :, :, 0]

        loss = self.cross_entropy_loss(downsampled_patches_node, predict_logits)

        self._train, self._learning_rate = self.optimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._patches_node = patches_node
        self._downsampled_patches_node = downsampled_patches_node
        self._predict_logits = predict_logits

        self._summary.initialize_train_summary()
        self._summary.initialize_missclassification_summary()

        summary_scalars = {"loss": loss, "learning_rate": self._learning_rate}
        self.summary_op = self._summary.get_summary_op(summary_scalars)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=100)

    def train(self, patches, downsampled_patches, train_images, downsampled_train_images):
        """Train the model for one epoch

        params:
            patches: [num_patches, patch_height, patch_width]
            imgs: [num_images, img_height, img_width]
        """
        opts = self._options

        num_train_patches = patches.shape[0]

        indices = np.arange(0, num_train_patches)
        np.random.shuffle(indices)

        num_errors = 0
        total = 0

        for batch_i, offset in enumerate(range(0, num_train_patches - opts.batch_size, opts.batch_size)):
            batch_indices = indices[offset:offset + opts.batch_size]

            feed_dict = {
                self._patches_node: patches[batch_indices, :, :],
                self._downsampled_patches_node: downsampled_patches[batch_indices, :, :],
                self._dropout_keep: opts.dropout,
            }

            summary_str, _, l, predictions, predictions, step = self._session.run([
                self.summary_op,
                self._train,
                self._loss,
                self._predict_logits,
                self._predictions,
                self._global_step
            ],feed_dict=feed_dict)

            print("Batch {} Step {}".format(batch_i, step), end="\r")
            self._summary.add(summary_str, global_step=step)

            num_errors += np.abs(patches[batch_indices] - predictions).sum()
            total += opts.batch_size
            self._summary.add_to_pixel_missclassification_summary(num_errors, total, self._global_step)

            if step > 0 and step % opts.train_score_every == 0:
                self._summary.add_to_training_summary(self.predict(downsampled_train_images), train_images, self._global_step)

        self._summary.flush()

    def predict(self, imgs):
        """Run inference on `imgs` and return predicted masks

        imgs: [num_images, image_height, image_width, num_channel]
        returns: masks [num_images, images_height, image_width] with road probabilities
        """
        opts = self._options

        num_images = imgs.shape[0]
        print("Running prediction on {} images... ".format(num_images), end="")

    def save(self, epoch=0):
        opts = self._options
        model_data_dir = os.path.abspath(
            os.path.join(opts.save_path, self.experiment_name, 'model-epoch-{:03d}.chkpt'.format(epoch)))
        saved_path = self.saver.save(self._session, model_data_dir)
        # create checkpoint
        print("Model saved in file: {}".format(saved_path))


def main(_):
    """

    """
    opts = Options()

    if opts.gpu == -1:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)

    with tf.Graph().as_default(), tf.Session(config=config) as session:
        device = '/device:CPU:0' if opts.gpu == -1 else '/device:GPU:{}'.format(opts.gpu)
        print("Running on device {}".format(device))
        with tf.device(device):
            model = ConvolutionalModel(opts, session)

        if opts.num_epoch > 0:
            # train model
            # TODO: make path to data a flag
            train_images = images.load_data("../data/Membrane_.tif")
            input_size = unet.input_size_needed(opts.patch_size, opts.num_layers)
            offset = int((input_size - opts.patch_size) / 2)

            input_size = unet.input_size_needed(opts.patch_size, opts.num_layers)
            patches = images.extract_patches(train_images, input_size, opts.stride)

            print("Train on {} patches of size {}x{}".format(
                patches.shape[0],
                patches.shape[1],
                patches.shape[2]
            ))

            downsampled_patches = images.extract_patches(train_images, opts.patch_size, opts.stride)
            downsampled_patches = images.downsample_patches(downsampled_patches)

            print("Train on {} downsampled patches of size {}x{}".format(
                downsampled_patches.shape[0],
                downsampled_patches.shape[1],
                downsampled_patches.shape[2]
            ))

            downsampled_train_images = images.downsample_patches(train_images)

            for i in range(opts.num_epoch):
                print("==== Train epoch: {} ====".format(i))
                # Reset scores
                tf.local_variables_initializer().run()
                # Process one epoch
                model.train(patches, downsampled_patches, train_images, downsampled_train_images)
                # TODO: Save model to disk
                model.save(i)

if __name__ == '__main__':
    tf.app.run()
