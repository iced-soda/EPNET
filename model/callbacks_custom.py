import logging
import warnings
import numpy as np
import pandas as pd
from keras.callbacks import Callback


# http://alexadam.ca/ml/2018/08/03/early-stopping.html
class FixedEarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitors: quantities to be monitored.
        min_deltas: minimum change in the monitored quantities
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        modes: list of {auto, min, max}. In `min` mode,
            training will stop when the quantities
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baselines: Baseline values for the monitored quantities to reach.
            Training will stop if the model doesn't show improvement
            for at least one of the baselines.
    """

    def __init__(self,
                 monitors=['val_loss'],
                 min_deltas=[0],
                 patience=0,
                 verbose=0,
                 modes=['auto'],
                 baselines=[None]):
        super(FixedEarlyStopping, self).__init__()

        self.monitors = monitors
        self.baselines = baselines
        self.patience = patience
        self.verbose = verbose
        self.min_deltas = min_deltas
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_ops = []

        for i, mode in enumerate(modes):
            if mode not in ['auto', 'min', 'max']:
                warnings.warn('EarlyStopping mode %s is unknown, '
                              'fallback to auto mode.' % mode,
                              RuntimeWarning)
                modes[i] = 'auto'

        for i, mode in enumerate(modes):
            if mode == 'min':
                self.monitor_ops.append(np.less)
            elif mode == 'max':
                # self.monitor_ops.append(np.greater)
                self.monitor_ops.append(np.greater_equal)
            else:
                if 'acc' in self.monitors[i]:
                    self.monitor_ops.append(np.greater)

                else:
                    self.monitor_ops.append(np.less)

        for i, monitor_op in enumerate(self.monitor_ops):
            if monitor_op == np.greater:
                self.min_deltas[i] *= 1
            else:
                self.min_deltas[i] *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.waits = []
        self.stopped_epoch = 0
        self.bests = []

        for i, baseline in enumerate(self.baselines):
            if baseline is not None:
                self.bests.append(baseline)
            else:
                self.bests.append(np.Inf if self.monitor_ops[i] == np.less else -np.Inf)

            self.waits.append(0)

    def on_epoch_end(self, epoch, logs=None):
        reset_all_waits = False
        for i, monitor in enumerate(self.monitors):
            current = logs.get(monitor)

            if current is None:
                warnings.warn(
                    'Early stopping conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (monitor, ','.join(list(logs.keys()))), RuntimeWarning
                )
                return

            if self.monitor_ops[i](current - self.min_deltas[i], self.bests[i]):
                self.bests[i] = current
                self.waits[i] = 0
                reset_all_waits = True
            else:
                self.waits[i] += 1

        if reset_all_waits:
            for i in range(len(self.waits)):
                self.waits[i] = 0

            return

        num_sat = 0
        for wait in self.waits:
            if wait >= self.patience:
                num_sat += 1

        if num_sat == len(self.waits):
            self.stopped_epoch = epoch
            self.model.stop_training = True

        print((self.waits))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(('Epoch %05d: early stopping' % (self.stopped_epoch + 1)))


class GradientCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, gradient_function, x_train, y_train, max_epoch, feature_names=None, monitor='val_loss',
                 verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=10):
        super(GradientCheckpoint, self).__init__()
        self.monitor = monitor
        self.feature_names = feature_names
        self.gradient_function = gradient_function
        self.x_train = x_train
        self.y_train = y_train
        self.verbose = verbose
        n = len(self.feature_names)
        self.history = [[] for i in range(n)]
        self.max_epoch = max_epoch
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        is_last_epoch = (self.max_epoch - epoch - 1) < self.period
        # is_last_epoch = self.max_epoch == epoch
        print(('is_last_epoch', is_last_epoch))
        if (self.epochs_since_last_save >= self.period) or (epoch == 0):
            self.epochs_since_last_save = 0
            logging.info('getting gradient')
            coef_ = self.gradient_function(self.model, self.x_train, self.y_train)
            if type(coef_) != list:
                coef_ = [coef_]
            # for i, c in enumerate(coef_):
            #     df = pd.DataFrame(c)
            #     logging.info('saving gradient epoch {} layer {}'.format(epoch, i))
            #     f= '{} epoch {} layer {} .csv'.format(self.filepath , str(epoch), str(i) )
            #     df.to_csv(f)
            i = 0

            for c, names in zip(coef_, self.feature_names):
                print(i)
                print((c.shape))
                print((len(names)))
                df = pd.DataFrame(c.ravel(), index=names, columns=[str(epoch)])
                self.history[i].append(df)
                # logging.info('saving gradient epoch {} layer {}'.format(epoch, i))
                # f= '{} epoch {} layer {} .csv'.format(self.filepath , str(epoch), str(i) )
                # df.to_csv(f)
                i += 1

            # save
            if is_last_epoch:
                logging.info('saving gradient')
                for i, h in enumerate(self.history):
                    df = pd.concat(h, axis=1)
                    f = '{} layer {} .csv'.format(self.filepath, str(i))
                    df.to_csv(f)

import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class MetricsCallback(Callback):
    def __init__(self, x_val, y_val):
        super(MetricsCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.metrics_history = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        self.csv_filename = 'metrics_log.csv'
        
    def on_epoch_end(self, epoch, logs=None):
        # Evaluate the model on the validation set
        y_pred = self.model.predict(self.x_val)

        # Calculate the average along the first axis (axis=0)
        y_pred_average = np.mean(y_pred, axis=0)

        y_pred_binary = (y_pred_average > 0.5).astype(int)

        # debugging
        # print("y_pred_average shape:", y_pred_average.shape)
        # print("x_val shape:", self.x_val.shape)
        # print("y_pred shape:", y_pred.shape)
        # print("Epoch:", epoch)
        # print("y_val shape:", self.y_val.shape)
        # print("y_val type:", type(self.y_val))
        # print("y_pred_binary shape:", y_pred_binary.shape)
        # print("y_pred_binary type:", type(y_pred_binary))
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val,y_pred_binary)
        precision = precision_score(self.y_val, y_pred_binary)
        recall = recall_score(self.y_val, y_pred_binary)
        f1 = f1_score(self.y_val, y_pred_binary)
        
        # Append metrics to history
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1'].append(f1)