"""

"""
import tensorflow as tf
import numpy as np

import unet
import images

from summary import Summary
from datetime import datetime

import os

import logging
logging.getLogger('tensorflow').disabled = True

# Set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.WARN)

# 
DEFAULT_FULL_PREDICTION = False
DEFAULT_GPU =  0
DEFAULT_BATCH_SIZE = 10
DEFAULT_PATCH_SIZE = 120
DEFAULT_STRIDE = 60
DEFAULT_SEED = 2018
DEFAULT_ROOT_SIZE = 16
DEFAULT_NUM_EPOCH = 40
DEFAULT_NUM_LAYERS = 3
DEFAULT_K_FACTOR = 3
DEFAULT_DILATION_SIZE = 3
DEFAULT_CONV_SIZE = 3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.99
DEFAULT_LOGDIR = os.path.abspath("./logdir")
DEFAULT_SAVE_PATH = os.path.abspath("./runs")
DEFAULT_DATA = os.path.abspath("../data/Membrane_.tif")
DEFAULT_LOG_SUFFIX = ""

tf.app.flags.DEFINE_boolean('full_prediction', DEFAULT_FULL_PREDICTION, "Whether or not to run a full volume prediction after training")

tf.app.flags.DEFINE_integer('gpu', DEFAULT_GPU, "GPU to run the model on")
tf.app.flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE, "Batch size of training instances")
tf.app.flags.DEFINE_integer('patch_size', DEFAULT_PATCH_SIZE, "Size of the prediction image")
tf.app.flags.DEFINE_integer('stride', DEFAULT_STRIDE, "Sliding delta for patches")
tf.app.flags.DEFINE_integer('seed', DEFAULT_SEED, "Random seed for reproducibility")
tf.app.flags.DEFINE_integer('root_size', DEFAULT_ROOT_SIZE, "Number of filters of the first U-Net layer")
tf.app.flags.DEFINE_integer('num_epoch', DEFAULT_NUM_EPOCH, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('num_layers', DEFAULT_NUM_LAYERS, "Number of layers of the U-Net")
tf.app.flags.DEFINE_integer('k_factor', DEFAULT_K_FACTOR, "Determines the factor by which training images are downsampled for training")
tf.app.flags.DEFINE_integer('dilation_size', DEFAULT_DILATION_SIZE, "Filter size of dilated convolution layer")
tf.app.flags.DEFINE_integer('conv_size', DEFAULT_CONV_SIZE, "Filter size of convolution layer")

tf.app.flags.DEFINE_float('learning_rate', DEFAULT_LEARNING_RATE, "Initial learning rate")
tf.app.flags.DEFINE_float('dropout', DEFAULT_DROPOUT, "Probability to keep an input")

tf.app.flags.DEFINE_string('logdir', DEFAULT_LOGDIR, "Directory where to write logfiles")
tf.app.flags.DEFINE_string('save_path', DEFAULT_SAVE_PATH, "Directory where to write checkpoints")
tf.app.flags.DEFINE_string('data', DEFAULT_DATA, "Data to learn on")
tf.app.flags.DEFINE_string('log_suffix', DEFAULT_LOG_SUFFIX, "suffix to attach to log folder")

FLAGS = tf.app.flags.FLAGS

class Options(object):
    """
        Options used by model
    """
    def __init__(self):
        self.full_prediction = FLAGS.full_prediction
        self.batch_size = FLAGS.batch_size
        self.gpu = FLAGS.gpu
        self.learning_rate = FLAGS.learning_rate
        self.logdir = FLAGS.logdir
        self.patch_size = FLAGS.patch_size
        self.save_path = FLAGS.save_path
        self.seed = FLAGS.seed
        self.stride = FLAGS.stride
        self.dropout = FLAGS.dropout
        self.dilation_size = FLAGS.dilation_size
        self.conv_size = FLAGS.conv_size
        self.root_size = FLAGS.root_size
        self.num_epoch = FLAGS.num_epoch
        self.num_layers = FLAGS.num_layers
        self.downsample_factor = FLAGS.k_factor
        self.data = FLAGS.data
        self.log_suffix = FLAGS.log_suffix

        if FLAGS.learning_rate != DEFAULT_LEARNING_RATE:
            self.log_suffix += "-lr_" + str(FLAGS.learning_rate)
        if FLAGS.patch_size != DEFAULT_PATCH_SIZE:
            self.log_suffix += "-ps_" + str(FLAGS.patch_size)
        if FLAGS.seed != DEFAULT_SEED:
            self.log_suffix += "-se_" + str(FLAGS.seed)
        if FLAGS.stride != DEFAULT_STRIDE:
            self.log_suffix += "-st_" + str(FLAGS.stride)
        if FLAGS.dropout != DEFAULT_DROPOUT:
            self.log_suffix += "-dr_" + str(FLAGS.dropout)
        if FLAGS.dilation_size != DEFAULT_DILATION_SIZE:
            self.log_suffix += "-di_" + str(FLAGS.dilation_size)
        if FLAGS.conv_size != DEFAULT_CONV_SIZE:
            self.log_suffix += "-co_" + str(FLAGS.conv_size)
        if FLAGS.root_size != DEFAULT_ROOT_SIZE:
            self.log_suffix += "-ro_" + str(FLAGS.root_size)
        if FLAGS.num_epoch != DEFAULT_NUM_EPOCH:
            self.log_suffix += "-ep_" + str(FLAGS.num_epoch)
        if FLAGS.num_layers != DEFAULT_NUM_LAYERS:
            self.log_suffix += "-la_" + str(FLAGS.num_layers)
        if FLAGS.k_factor != DEFAULT_K_FACTOR:
            self.log_suffix += "-k_" + str(FLAGS.k_factor)

class ConvolutionalModel:
    def __init__(self, options, session):
        self._options = options
        self._session = session

        self.train_images_shape = None

        np.random.seed(options.seed)
        tf.set_random_seed(options.seed)
        self.input_size = self._options.patch_size

        self.experiment_name = datetime.now().strftime("%Y%m%d%H%M%S")
        experiment_path = os.path.abspath(os.path.join(options.save_path, self.experiment_name))
        self.summary_path = os.path.join(options.logdir, self.experiment_name + options.log_suffix)

        self._summary = Summary(options, session)
        self.build_graph()

    def calculate_loss_abs(self, labels, prediction):
        """Calculate absolute difference loss

        """
        loss = tf.losses.absolute_difference(
            labels,
            prediction
        )

        return loss

    def calculate_loss_mse(self, labels, prediction):
        """Calculate mean squared error loss

        """
        loss = tf.losses.mean_squared_error(
            labels,
            prediction
        )

        return loss

    def calculate_loss_snr(self, labels, prediction):
        """Calculate loss based on signal to noise
        """
        loss = tf.negative(
            tf.multiply(
                tf.constant(20.0),
                tf.subtract(
                    self.tf_log_10(self.tf_range(labels)),
                    self.tf_log_10(tf.sqrt(tf.losses.mean_squared_error(labels, prediction)))
                )
            ),
            name="snr"
        )
        print(loss)

        return loss

    def tf_log_10(self, x):
        """ log10 implemented using tensorflow
        """
        return tf.divide(tf.log(x), tf.log(tf.constant(10.0)))

    def tf_range(self, img):
        """ calculate dynamic range of an image using tensorflow
        """
        return tf.subtract(tf.reduce_max(img), tf.reduce_min(img))

    def optimize(self, loss):
        """optimize with MomentumOptimizer

        """
        learning_rate = tf.train.exponential_decay(
            self._options.learning_rate,
            self._global_step,
            100,
            0.99,
            staircase=True
        )

        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        train = optimizer.minimize(loss, global_step=self._global_step)

        return train, learning_rate

    def adam_optimize(self, loss):
        """ optimize with AdamOptimizer
        """
        optimizer = tf.train.AdamOptimizer(self._options.learning_rate, epsilon=10e-3)
        train = optimizer.minimize(loss, global_step=self._global_step)

        return train, optimizer._lr

    def build_graph(self):
        """
            Build the tensorflow graph for the model
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
                self.input_size,
                1
            ),
            name="patches"
        )

        labels_node = tf.placeholder(
            tf.float32,
            shape=(
                self._options.batch_size,
                self._options.patch_size,
                self._options.patch_size,
                1
            ),
            name="labels"
        )

        dropout_keep = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep")
        self._dropout_keep = dropout_keep

        print("Patches node: {}".format(patches_node))
        predict_logits = unet.forward(
            patches_node,
            root_size=opts.root_size,
            num_layers=opts.num_layers,
            dropout_keep=dropout_keep,
            dilation_size=opts.dilation_size,
            conv_size=opts.conv_size
        )

        predictions = predict_logits

        print("Predicted logits: {}".format(predict_logits))
        loss = self.calculate_loss_snr(labels_node, predict_logits)

        self._train, self._learning_rate = self.adam_optimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._patches_node = patches_node
        self._labels_node = labels_node
        self._predict_logits = predict_logits

        self._summary.create_writer(self.summary_path)
        self._summary.initialize_snr_summary()
        self._summary.initialize_eval_summary()

        summary_scalars = {"loss": loss, "learning_rate": self._learning_rate}
        self.summary_op = self._summary.get_summary_op(summary_scalars)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=100)

    def train(self, patches, labels_patches, eval_images, downsampled_eval_images):
        """Train the model for one epoch

        params:
            patches: [num_patches, patch_height, patch_width, num_channel]
            labels_patches: [num_patches, patch_height, patch_width, num_channel]
            eval_images: [num_images, img_height, img_width, num_channel]
            downsampled_eval_images: [num_images, img_height, img_width, num_channel]
        """
        opts = self._options

        num_train_patches = patches.shape[0]

        indices = np.arange(0, num_train_patches)

        # randomize indices for training
        np.random.shuffle(indices)

        for batch_i, offset in enumerate(range(0, num_train_patches - opts.batch_size, opts.batch_size)):
            batch_indices = indices[offset:offset + opts.batch_size]

            feed_dict = {
                self._patches_node: patches[batch_indices, :, :, :],
                self._labels_node: labels_patches[batch_indices, :, :, :],
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

            snr = images.psnr(labels_patches[batch_indices], predictions)
            self._summary.add_to_snr_summary(snr, self._global_step)

            # Do evaluation once per epoch
            if step > 0 and step % int(patches.shape[0] / opts.batch_size) == 0:
                predictions = self.predict(downsampled_eval_images)

                self._summary.add_to_eval_summary(
                    eval_images,
                    downsampled_eval_images,
                    predictions,
                    self._global_step
                )

        self._summary.flush()

    def predict(self, imgs):
        """Run inference on `imgs` and return predictions

        imgs: [num_images, image_height, image_width, num_channel]
        returns: predictions [num_images, images_height, image_width, num_channel]
        """
        opts = self._options

        num_images = imgs.shape[0]

        print()
        print("Running prediction on {} images with shape {}... ".format(num_images, imgs.shape))

        patches = tf.extract_image_patches(
            imgs,
            [1, self.input_size, self.input_size, 1],
            [1, opts.stride, opts.stride, 1],
            [1, 1, 1, 1],
            'VALID'
        ).eval()
        patches = patches.reshape((-1, self.input_size, self.input_size, 1))

        num_patches = patches.shape[0]

        # patches padding to have full batches
        if num_patches % opts.batch_size != 0:
            num_extra_patches = opts.batch_size - (num_patches % opts.batch_size)
            extra_patches = np.zeros((num_extra_patches, self.input_size, self.input_size, 1))
            patches = np.concatenate([patches, extra_patches], axis=0)

        num_patches = patches.shape[0]

        num_batches = int(num_patches / opts.batch_size)
        eval_predictions = np.ndarray(shape=(num_patches, opts.patch_size, opts.patch_size, 1))
        print("Patches to predict: ", num_patches)
        print("Shape eval predictions: ", eval_predictions.shape)

        # do batchwise prediction
        for batch in range(num_batches):
            offset = batch * opts.batch_size

            feed_dict = {
                self._patches_node: patches[offset:offset + opts.batch_size, :, :, :],
            }
            eval_predictions[offset:offset + opts.batch_size, :, :, :] = self._session.run(self._predictions, feed_dict)

        # remove padding
        eval_predictions = eval_predictions[0:num_patches]


        # construct predicted images
        predictions = images.images_from_patches(eval_predictions, imgs.shape, stride=opts.stride)

        # Clipping for display in tensorboard
        predictions[predictions < 0] = 0
        predictions[predictions > 1] = 1

        return predictions


    def save(self, epoch=0):
        """
            Saves the training state of the model to disk to continue training at a later point
        """
        opts = self._options
        model_data_dir = os.path.abspath(
            os.path.join(opts.save_path, self.experiment_name, 'model-epoch-{:03d}.chkpt'.format(epoch)))
        saved_path = self.saver.save(self._session, model_data_dir)
        # create checkpoint
        print("Model saved in file: {}".format(saved_path))


def main(_):
    """
        main routine that initializes and launches the training process according
        to the parameters supplied as flags when executing this file.
    """
    opts = Options()

    if opts.gpu == -1:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Graph().as_default(), tf.Session(config=config) as session:
        device = '/device:CPU:0' if opts.gpu == -1 else '/device:GPU:{}'.format(opts.gpu)
        print("Running on device {}".format(device))
        with tf.device(device):
            # Initialize Model
            model = ConvolutionalModel(opts, session)

            # Load training data
            train_images = images.load_data(os.path.abspath(opts.data))

            # cut image to patch friendly size
            size_x = int(train_images.shape[1] / opts.stride) * opts.stride
            size_y = int(train_images.shape[2] / opts.stride) * opts.stride
            start_x = int((train_images.shape[1]-size_x)/2)
            start_y = int((train_images.shape[2]-size_y)/2)

            train_images = train_images[:,start_x:start_x+size_x,start_y:start_y+size_y,:]

            model.train_images_shape = train_images.shape

            # Eval images
            eval_images = np.swapaxes(train_images, 0, 1)
            downsampled_eval_images = images.downsample(eval_images, opts.downsample_factor, get_all_patches=False)
            downsampled_eval_images = tf.image.resize_bicubic(
                downsampled_eval_images,
                np.array([model.train_images_shape[1], model.train_images_shape[2]]),
                align_corners=True
            )
            downsampled_eval_images = downsampled_eval_images.eval()

        # Start training
        if opts.num_epoch > 0:
            # train model

            labels_patches = tf.extract_image_patches(
                train_images,
                [1, model.input_size, model.input_size, 1],
                [1, opts.stride, opts.stride, 1],
                [1, 1, 1, 1],
                'VALID'
            ).eval()
            labels_patches = labels_patches.reshape((-1, model.input_size, model.input_size, 1))
            print("Shape labels_patches: ", labels_patches.shape)

            # downsample patches and resize them back to original dimensions
            patches = images.downsample(labels_patches, opts.downsample_factor)
            patches = tf.image.resize_images(
                patches,
                np.array([model.input_size, model.input_size]),
                method=tf.image.ResizeMethod.BICUBIC,
                align_corners=True
            )

            # resize labels_patches to compensate for downsampling
            labels_patches = np.tile(labels_patches, (opts.downsample_factor, 1, 1, 1))

            print("Train on {} labels_patches of size {}x{}".format(
                labels_patches.shape[0],
                labels_patches.shape[1],
                labels_patches.shape[2]
            ))

            print("Train on {} patches of size {}x{}".format(
                patches.shape[0],
                patches.shape[1],
                patches.shape[2]
            ))



            for i in range(opts.num_epoch):
                print("==== Train epoch: {} ====".format(i))
                # Reset scores
                tf.local_variables_initializer().run()
                # Process one epoch
                eval_ids = int(eval_images.shape[0] / 2)
                print("Eval on image: {}".format(eval_ids))
                model.train(patches.eval(), labels_patches, eval_images[eval_ids:eval_ids+1], downsampled_eval_images[eval_ids:eval_ids+1])
                memop = tf.contrib.memory_stats.MaxBytesInUse()
                print("Memory in use {:.2f} GB".format(memop.eval()/10**9))
            print("Training finished")

        # Do a full prediction after training if flag is passed
        if opts.full_prediction:
            print("Generating full prediction on {}".format(opts.data))
            full_pred = model.predict(np.swapaxes(downsampled_eval_images, 0, 1))

            # Clip images for display
            downsampled_eval_images[downsampled_eval_images < 0] = 0
            downsampled_eval_images[downsampled_eval_images > 1] = 1

            full_pred[full_pred < 0] = 0
            full_pred[full_pred > 1] = 1

            np.save(os.path.abspath(os.path.join(opts.save_path, "bicubic.npy")), downsampled_eval_images)
            np.save(os.path.abspath(os.path.join(opts.save_path, "full_prediction.npy")), full_pred)

            images.save_array_as_tif(downsampled_eval_images, os.path.abspath(os.path.join(opts.save_path, "bicubic.tif")))
            images.save_array_as_tif(full_pred, os.path.abspath(os.path.join(opts.save_path, "full_prediction.tif")))
            print("Predictions saved")


if __name__ == '__main__':
    tf.app.run()
