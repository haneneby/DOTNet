import logging
import os
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
import matplotlib
import matplotlib.pyplot as plt
from Utils.Utils_models import *
from Utils.visualize import *

matplotlib.use('Agg')
lgr = None

def initlogger(configuration):
    global lgr
    if lgr is None:
        lgr = logging.getLogger('global')
    if 'logdir' in configuration:
        fh = logging.FileHandler(os.path.join(configuration['logdir'], 'MultiNet.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
        fh.setFormatter(formatter)
        lgr.addHandler(fh)
    lgr.setLevel(logging.INFO)
    return lgr


def getlogger():
    global lgr
    if lgr is None:
        return initlogger({})
    return lgr

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        # clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model loss')
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("Output/modelloss.png", bbox_inches='tight')
        plt.close("all")

plot_losses = PlotLosses()

class MyCallback(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch >10 and K.get_value(self.alpha)<0.4:
            # K.set_value(self.alpha, 0.4)
            # K.set_value(self.beta, K.get_value(self.beta) +0.0001)
#             if  K.get_value(self.alpha)<1: #was 0.3
                 K.set_value(self.alpha, K.get_value(self.alpha) +0.001)
# #            K.set_value(self.alpha, max(0.75, K.get_value(self.alpha) -0.0001))
# #                  K.set_value(self.beta,  min(0.7, K.get_value(self.beta) -0.0001))
        logger.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=30):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) 

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()# K.get_value(model.optimizer.lr)

    new_lr = old_lr * 0.99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)
    # if epoch >= 1: 
    #     plot_generated_images(epoch, self.model, True,Tmp_ssimlist)
    #     plot_confusionmatrix(epoch, self.model)
    #     plot_roc_curve(self.model)

    PlotLosses()
    if epoch == 5 or (epoch >= 10 and epoch % 10 == 0):
        self.model.save('./Output/checkpoint.h5')
        # plot_generated_images(dir, Im_pred_1, x_test, True)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=30):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)  




