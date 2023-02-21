import numpy as np
import h5py
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse, os, time, logging
from tqdm import tqdm
import pandas as pd
import multiprocessing
from functools import partial
import pickle
from model import UNet, ModelConfig
from data_reader import DataReader
from data_reader import generator
from postprocess import extract_picks, save_picks, save_picks_json, extract_amplitude, convert_true_picks, calc_performance
from visulization import plot_waveform
from util import EMA, LMA

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", help="train/train_valid/test/debug")
    parser.add_argument("--epochs", default=5, type=int, help="number of epochs (default: 10)")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--drop_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--decay_step", default=-1, type=int, help="decay step")
    parser.add_argument("--decay_rate", default=0.9, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--optimizer", default="adam", help="optimizer: adam, momentum")
    parser.add_argument("--summary", default=True, type=bool, help="summary")
    parser.add_argument("--class_weights", nargs="+", default=[1, 1, 1], type=float, help="class weights")
    parser.add_argument("--model_dir", default='log/230216-160012/models', help="Checkpoint directory (default: None)")
    parser.add_argument("--load_model", action="store_true", help="Load checkpoint")
    parser.add_argument("--log_dir", default="log", help="Log directory (default: log)")
    parser.add_argument("--num_plots", default=10, type=int, help="Plotting training results")
    parser.add_argument("--min_p_prob", default=0.3, type=float, help="Probability threshold for P pick")
    parser.add_argument("--min_s_prob", default=0.3, type=float, help="Probability threshold for S pick")
    parser.add_argument("--format", default="numpy", help="Input data format")
    parser.add_argument("--train_dir", default="./dataset/waveform_train/", help="Input file directory")
    parser.add_argument("--train_list", default="./dataset/waveform.csv", help="Input csv file")
    parser.add_argument("--valid_dir", default=None, help="Input file directory")
    parser.add_argument("--valid_list", default=None, help="Input csv file")
    parser.add_argument("--test_dir", default=None, help="Input file directory")
    parser.add_argument("--test_list", default=None, help="Input csv file")
    parser.add_argument("--result_dir", default="results", help="result directory")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--save_prob", action="store_true", help="If save result for test")
    args = parser.parse_args()

    return args


def test(args, data_reader):

    current_time = time.strftime("%y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.info("Training log: {}".format(log_dir))
    model_dir = os.path.join(log_dir, 'models')
    os.makedirs(model_dir)

    figure_dir = os.path.join(log_dir, 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    config = ModelConfig(X_shape=[3001, 1, 3], Y_shape=[3001, 1, 3])
    if args.decay_step == -1:
        args.decay_step = data_reader.num_data // args.batch_size
    config.update_args(args)
    with open(os.path.join(log_dir, 'config.log'), 'w') as fp:
        fp.write('\n'.join("%s: %s" % item for item in vars(config).items()))

    with tf.compat.v1.name_scope('Input_Batch'):
        dataset = data_reader.batch(args.batch_size, drop_remainder=True)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        #print(batch.shape)

    model = UNet(config, input_batch=batch)
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # sess_config.log_device_placement = False

    with tf.compat.v1.Session(config=sess_config) as sess:

        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        if args.model_dir is not None:
            logging.info("restoring models...")
            latest_check_point = tf.train.latest_checkpoint(args.model_dir)
            saver.restore(sess, latest_check_point)

        if args.plot_figure:
            multiprocessing.set_start_method('spawn')
            pool = multiprocessing.Pool(multiprocessing.cpu_count())

        flog = open(os.path.join(log_dir, 'loss.log'), 'w')
        test_loss = LMA()
        results = {'wfs': [], 'truth': [], 'preds': []}
        progressbar = tqdm(range(0, data_reader.num_data, args.batch_size), desc="{}: ".format(log_dir.split("/")[-1]))
        for _ in progressbar:
            loss_batch, preds_batch, X_batch, Y_batch = sess.run([model.loss, model.preds, batch[0], batch[1]],
                                                                              feed_dict={model.drop_rate: 0, model.is_training: False})
            test_loss(loss_batch)
            progressbar.set_description("{}: loss={:.6f}, mean={:.6f}".format(log_dir.split("/")[-1], loss_batch, test_loss.value))
            results['wfs'].append(X_batch)
            results['truth'].append(Y_batch)
            results['preds'].append(preds_batch)


        flog.write("mean loss: {}\n".format(test_loss.value))
        flog.flush()

        flog.close()

    return 0

def main(args):

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    coord = tf.train.Coordinator()

    data_pwd = "/scratch/alpine/mecr8410/ML_seismo/wavelets/scripts/wave/dataset/subset.hdf5"
    #data_pwd = "/scratch/alpine/mecr8410/ML_seismo/wavelets/scripts/wave/dataset/ds_wf.hdf5"
    f = h5py.File(data_pwd, 'r')

    with tf.compat.v1.name_scope('create_inputs'):
            data_reader = tf.data.Dataset.from_generator(
                                            generator(data_pwd, 'test'),
                                            output_signature = (tf.TensorSpec(shape=(3001,1,3), dtype=tf.float32),
                                            tf.TensorSpec(shape=(3001,1,3), dtype=tf.float32)))
            data_reader.num_data = len(f['test']['X'])
            f.close()
            logging.info("Dataset size: test {}".format(data_reader.num_data))
    test(args, data_reader)

    return


if __name__ == '__main__':
    args = read_args()
    main(args)
