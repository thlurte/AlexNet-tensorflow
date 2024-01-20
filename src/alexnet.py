import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfl
import tensorflow.nn as tfn

class AlexNet:
  def __init__(self,input_size,batch_size):

    # Image Input Size
    self.input_size=input_size


    # Batch Size
    self.batch_size=batch_size

    # Hyperparameters
    # Filters
    self.filters={
        "L1":96

        ,"L2":256

        ,"L3":384

        ,"L4":384

        ,"L5":256
        }

    # Kernel Size
    self.kernel_size={
        "L1":[11,11]

        ,"L2":[5,5]

        ,"L3":[3,3]

        ,"L4":[3,3]

        ,"L5":[3,3]
        }

    # Fully Connected Layers
    self.fc={
        "L6":4096

        ,"L7":4096

        ,"L8":1000
        }

    # Stirdes
    self.strides={
        "L1":4
        }

    # Padding
    self.padding={
        "L1":"valid"

        ,"L2":"same"

        ,"L3":"same"

        ,"L4":"same"

        ,"L5":"same"
        }

    # Maxpool
    self.maxpool_strides={
        "L1":2

        ,"L2":2

        ,"L5":2
        }

    self.maxpool_filter_size={
        "L1":[3,3]

        ,"L2":[3,3]

        ,"L5":[3,3]
        }

    # Initial Weights
    self.initial_weights=tfk.initializers.RandomNormal(mean=0.0,stddev=0.01)

  def model(self):
    #-----------------------------------#
    # Input Layer
    #-----------------------------------#

    input=tfl.Input(
        # Input shape of the image
        shape=self.input_size
        )

    #-----------------------------------#
    # Layer 01
    #-----------------------------------#
    x=tfl.Conv2D(
        # L1 Filters
        filters=self.filters["L1"]
        # L1 Kernel Size
        ,kernel_size=self.kernel_size["L1"]
        # L1 Strides
        ,strides=self.strides["L1"]
        # L1 Padding
        ,padding=self.padding["L1"]
        # L1 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L1 Bias Initializer
        ,bias_initializer=tf.constant_initializer(0)
    )(input)

    # L1 ReLU Activation
    x=tfl.Activation('relu')(x)

    # L1 Local Response Normalization
    x=tfl.Lambda(tfn.local_response_normalization)(x)

    # L1 Pooling Layer
    x=tfl.MaxPool2D(
        # Maxpool Filter Size
        pool_size=self.maxpool_filter_size["L1"]
        # Maxpool Strides
        ,strides=self.maxpool_strides["L1"]
        )(x)

    #-----------------------------------#
    # Layer 02
    #-----------------------------------#
    x=tfl.Conv2D(
        # L2 Filters
        filters=self.filters["L2"]
        # L2 Kernel Size
        ,kernel_size=self.kernel_size["L2"]
        # L2 Padding
        ,padding=self.padding["L2"]
        # L2 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L2 Bias Initializer
        ,bias_initializer=tf.constant_initializer(1)
    )(x)

    # L2 ReLU Activation
    x=tfl.Activation('relu')(x)

    # L2 Local Response Normalization
    x=tfl.Lambda(tfn.local_response_normalization)(x)

    # L2 Pooling Layer
    x=tfl.MaxPool2D(
        # Maxpool Filter Size
        pool_size=self.maxpool_filter_size["L2"]
        # Maxpool Strides
        ,strides=self.maxpool_strides["L2"]
        )(x)

    #-----------------------------------#
    # Layer 03
    #-----------------------------------#
    x=tfl.Conv2D(
        # L3 Filters
        filters=self.filters["L3"]
        # L3 Kernel Size
        ,kernel_size=self.kernel_size["L3"]
        # L3 Padding
        ,padding=self.padding["L3"]
        # L3 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L3 Bias Initializer
        ,bias_initializer=tf.constant_initializer(0)
    )(x)

    # L3 ReLU Activation
    x=tfl.Activation('relu')(x)

    #-----------------------------------#
    # Layer 04
    #-----------------------------------#
    x=tfl.Conv2D(
        # L4 Filters
        filters=self.filters["L4"]
        # L4 Kernel Size
        ,kernel_size=self.kernel_size["L4"]
        # L4 Padding
        ,padding=self.padding["L4"]
        # L4 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L4 Bias Initializer
        ,bias_initializer=tf.constant_initializer(1)
    )(x)

    # L4 ReLU Activation
    x=tfl.Activation('relu')(x)

    #-----------------------------------#
    # Layer 05
    #-----------------------------------#
    x=tfl.Conv2D(
        # L5 Filters
        filters=self.filters["L5"]
        # L5 Kernel Size
        ,kernel_size=self.kernel_size["L5"]
        # L5 Padding
        ,padding=self.padding["L5"]
        # L5 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L5 Bias Initializer
        ,bias_initializer=tf.constant_initializer(1)
    )(x)

    # L5 ReLU Activation
    x=tfl.Activation('relu')(x)

    # L5 Pooling Layer
    x=tfl.MaxPool2D(
        # Maxpool Filter Size
        pool_size=self.maxpool_filter_size["L5"]
        # Maxpool Strides
        ,strides=self.maxpool_strides["L5"]
        )(x)

    # Flatten
    x=tfl.Flatten()(x)

    #-----------------------------------#
    # Layer 06
    #-----------------------------------#
    x=tfl.Dense(
        # L6 Units
        units=4096
        # L6 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L6 Bias Initializer
        ,bias_initializer=tf.constant_initializer(1)
    )(x)

    # ReLU Activation
    x=tfl.Activation('relu')(x)

    # Dropout Layer
    x=tfl.Dropout(rate=0.5)(x)

    #-----------------------------------#
    # Layer 07
    #-----------------------------------#
    x=tfl.Dense(
        # L7 Units
        units=4096
        # L7 Kernel Initializer
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
                # L1 Bias Initializer
        ,bias_initializer=tf.constant_initializer(1)
    )(x)

    # ReLU Activation
    x=tfl.Activation('relu')(x)

    # Dropout Later
    x=tfl.Dropout(rate=0.5)(x)

    #-----------------------------------#
    # Layer 08
    #-----------------------------------#

    x=tfl.Dense(
        # L8 Units
        units=1000
        # L8 Kernel Units
        ,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        # L8 Bias Initializer
        ,bias_initializer=tf.constant_initializer(1)

    )(x)

    # Sigmoid Activation
    x=tfl.Activation('sigmoid')(x)

    return input,x


  def build(self,model,x):

    model=tfk.Model(input,x)

    model.compile(
        optimizer=tfk.optimizers.experimental.SGD(
            # Learning Rate
            learning_rate=0.01
            # Momentum
            ,momentum=0.9
            # Weigth Decay
            ,weight_decay=0.0005
        )
        # Loss Function
        ,loss='sparse_categorical_crossentropy'
        # Metrics
        ,metrics=['accuracy']
    )

    return model

  def fit(self,model,trainds,testds,epochs):

    history=model.fit(
        trainds
        ,batch_size=self.batch_size
        ,epochs=epochs
        ,validation_data=testds
        ,verbose=2
    )
    return history



