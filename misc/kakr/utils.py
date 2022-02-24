import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback

class Tensorboard_loss_viewer(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        train_log_dir = os.path.join(log_dir, 'train')
        super(Tensorboard_loss_viewer, self).__init__(train_log_dir, **kwargs)
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(Tensorboard_loss_viewer, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = 'epoch_'+ name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k:v for k,v in logs.items() if not k.startswith('val_')}
        super(Tensorboard_loss_viewer, self).on_epoch_end(epoch, logs)
    
    
# https://www.kaggle.com/c/2019-3rd-ml-month-with-kakr/discussion/103725#latest-597942    
class EpochLogWrite(Callback): # use only kaggle kernel
    from datetime import datetime
    from pytz import timezone, utc
    KST = timezone('Asia/Seoul')
    
    def on_epoch_begin(self, epoch, logs={}):
        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()
        print2('Epoch #{} begins at {}'.format(epoch+1, tmx))
    def on_epoch_end(self, epoch, logs={}):
        tmx = utc.localize(datetime.utcnow()).astimezone(KST).time()
        print2('Epoch #{} ends at {}  acc={} val_acc={} val_f1={}'.format(epoch+1, tmx, round(logs['acc'],4), round(logs['val_acc'],4), round(logs['val_f1_m'],4) ))


def print2(string): # use only kaggle kernel
  os.system(f'echo \"{string}\"')
  print(string)
