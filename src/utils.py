import tensoflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt


class Loader:
  def __init__(self,data_dir,img_size,batch_size):
    self.dir=dir
    self.img_size=img_size
    self.batch_size=batch_size

  def load(self):
    trainds=tfk.utils.image_dataset_from_directory(
        self.dir
        ,validation_split=0.04
        ,subset='training'
        ,seed=1218
        ,image_size=(self.img_size[0],self.img_size[1])
        ,batch_size=self.batch_size
    )
    testds=tfk.utils.image_dataset_from_directory(
        self.dir
        ,validation_split=0.04
        ,subset='validation'
        ,seed=1218
        ,image_size=(self.img_size[0],self.img_size[1])
        ,batch_size=self.batch_size
    )
    return trainds,testds


class Visualize:
  def __init__(self,history):
    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history.history['accuracy'])
    axs[1].plot(history.history['val_accuracy']) 
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    axs[0].figure.savefig("training_loss.png")
    axs[1].figure.savefig("training_accuracy.png")
