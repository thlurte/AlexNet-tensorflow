import tensoflow as tf
import tensorflow.keras as tfk

class Loader:
  def __init__(self,dir,img_size,batch_size):
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
