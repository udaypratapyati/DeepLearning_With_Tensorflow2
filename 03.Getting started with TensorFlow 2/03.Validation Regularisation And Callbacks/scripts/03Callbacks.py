#Callbacks are designed to monitor the loss in metrics at certain points in the training run
#and perform some action that depend on those loss in metric values.

from keras.callbacks import Callback

#class my_callback(Callback):
    
    #def on_train_begin(self, logs=None):
        # Do something at the start of the training
    
    #def on_train_batch_begin(self, batch, logs=None):
        # Do something at the start of the every batch iteration
    
    #def on_epoch_end(self, epoch, logs=None):
        # Do something at the end of every epoch

# history = model.fit(X_train, y_train, epochs=5, callbacks=[my_callback()])