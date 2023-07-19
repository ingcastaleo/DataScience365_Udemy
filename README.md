{
 "cells": [
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Loading libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import tensorflow_datasets as tfds\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set()\n"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Loading Mnist dataset and segmenting it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "mnist_dataset, mnist_info = tfds.load(name=\"mnist\", with_info=True, as_supervised=True)\n",
    "\n",
    "mnist_train, mnist_test = mnist_dataset[\"train\"], mnist_dataset[\"test\"]\n",
    "\n",
    "num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples # Get number of samples for validation (10% of training dataset)\n",
    "num_validation_samples = tf.cast(num_validation_samples,tf.int64)      # Cast the number of samples for validation as int64\n",
    "\n",
    "num_test_samples = mnist_info.splits['test'].num_examples  # Get number of samples for testing\n",
    "num_test_samples = tf.cast(num_test_samples,tf.int64)      # Cast the number of samples for testing as int64\n",
    "\n",
    "def scale(image,label):\n",
    "    image = tf.cast(image, tf.float32) # Cast data as float32\n",
    "    image /= 255.   # Divide over 255 and cast it as float32\n",
    "    return image, label\n",
    "\n",
    "scaled_train_and_validation_data = mnist_train.map(scale)\n",
    "scaled_test_data = mnist_test.map(scale)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Shuffle the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "BUFFER_SIZE = 10000 #Use at most 10000 samples at a time\n",
    "\n",
    "shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)\n",
    "\n",
    "validation_data = shuffled_train_and_validation_data.take(num_validation_samples)\n",
    "train_data = shuffled_train_and_validation_data.skip(num_validation_samples)"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Set the batch size\n",
    "\n",
    "\n",
    "Batch size = 1 --> Stochastic Gradient Descent (SGD)\n",
    "\n",
    "Batch size = #Samples --> Single Batch (GD)\n",
    "\n",
    "1 < Batch size < #Samples --> Mini-batch (GD)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 100\n",
    "\n",
    "train_data = train_data.batch(BATCH_SIZE)\n",
    "validation_data = validation_data.batch(num_validation_samples)\n",
    "test_data = scaled_test_data.batch(num_test_samples)\n",
    "\n",
    "validation_inputs, validation_targets = next(iter(validation_data))"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Outlining the module"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Choosing optimizer and loss function\n",
    "\n",
    "- binary_crossentropy --> Data with binary encoder\n",
    "- categorical_crossentropy --> Data has encoded with one-hot encoder\n",
    "- sparse_categorical_crossentropy --> Applies one-hot encoder "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "540/540 - 1s - loss: 0.4177 - accuracy: 0.8824 - val_loss: 0.2208 - val_accuracy: 0.9392 - 1s/epoch - 2ms/step\n",
      "Epoch 2/5\n",
      "540/540 - 1s - loss: 0.1904 - accuracy: 0.9449 - val_loss: 0.1561 - val_accuracy: 0.9553 - 669ms/epoch - 1ms/step\n",
      "Epoch 3/5\n",
      "540/540 - 1s - loss: 0.1454 - accuracy: 0.9563 - val_loss: 0.1178 - val_accuracy: 0.9670 - 663ms/epoch - 1ms/step\n",
      "Epoch 4/5\n",
      "540/540 - 1s - loss: 0.1160 - accuracy: 0.9656 - val_loss: 0.1035 - val_accuracy: 0.9708 - 669ms/epoch - 1ms/step\n",
      "Epoch 5/5\n",
      "540/540 - 1s - loss: 0.0965 - accuracy: 0.9714 - val_loss: 0.0879 - val_accuracy: 0.9737 - 664ms/epoch - 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27faf55c810>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem Higher Width (1000)\n",
    "\n",
    "The validation accuracy is marginally superior than the obtained with the former model. \n",
    "At the last epoch, it tends to overfit\n",
    "Significantly slower"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "540/540 - 8s - loss: 0.1945 - accuracy: 0.9407 - val_loss: 0.1020 - val_accuracy: 0.9702 - 8s/epoch - 14ms/step\n",
      "Epoch 2/5\n",
      "540/540 - 7s - loss: 0.0757 - accuracy: 0.9761 - val_loss: 0.0662 - val_accuracy: 0.9790 - 7s/epoch - 14ms/step\n",
      "Epoch 3/5\n",
      "540/540 - 7s - loss: 0.0500 - accuracy: 0.9838 - val_loss: 0.0507 - val_accuracy: 0.9850 - 7s/epoch - 13ms/step\n",
      "Epoch 4/5\n",
      "540/540 - 7s - loss: 0.0371 - accuracy: 0.9885 - val_loss: 0.0592 - val_accuracy: 0.9842 - 7s/epoch - 13ms/step\n",
      "Epoch 5/5\n",
      "540/540 - 7s - loss: 0.0317 - accuracy: 0.9900 - val_loss: 0.0543 - val_accuracy: 0.9825 - 7s/epoch - 14ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fb0e7cfd0>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 1000\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])\n",
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem Deeper (3 hidden layers)\n",
    "\n",
    "The validation accuracy is the same gotten with two hidden layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "540/540 - 1s - loss: 0.4117 - accuracy: 0.8803 - val_loss: 0.1900 - val_accuracy: 0.9485 - 1s/epoch - 3ms/step\n",
      "Epoch 2/5\n",
      "540/540 - 1s - loss: 0.1699 - accuracy: 0.9501 - val_loss: 0.1356 - val_accuracy: 0.9630 - 693ms/epoch - 1ms/step\n",
      "Epoch 3/5\n",
      "540/540 - 1s - loss: 0.1276 - accuracy: 0.9622 - val_loss: 0.1106 - val_accuracy: 0.9658 - 688ms/epoch - 1ms/step\n",
      "Epoch 4/5\n",
      "540/540 - 1s - loss: 0.1051 - accuracy: 0.9681 - val_loss: 0.1018 - val_accuracy: 0.9695 - 700ms/epoch - 1ms/step\n",
      "Epoch 5/5\n",
      "540/540 - 1s - loss: 0.0897 - accuracy: 0.9731 - val_loss: 0.0764 - val_accuracy: 0.9780 - 682ms/epoch - 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fbc38b750>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 2 to hidden layer 3\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 3 to Output layer \n",
    "                            ])\n",
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem with change in action function (sigmoid)\n",
    "\n",
    "Sigmoid gives a lower performance and slower increase in accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "540/540 - 1s - loss: 1.0307 - accuracy: 0.7672 - val_loss: 0.4464 - val_accuracy: 0.8918 - 1s/epoch - 2ms/step\n",
      "Epoch 2/5\n",
      "540/540 - 1s - loss: 0.3416 - accuracy: 0.9106 - val_loss: 0.2709 - val_accuracy: 0.9302 - 665ms/epoch - 1ms/step\n",
      "Epoch 3/5\n",
      "540/540 - 1s - loss: 0.2445 - accuracy: 0.9308 - val_loss: 0.2155 - val_accuracy: 0.9403 - 667ms/epoch - 1ms/step\n",
      "Epoch 4/5\n",
      "540/540 - 1s - loss: 0.2003 - accuracy: 0.9431 - val_loss: 0.1825 - val_accuracy: 0.9495 - 667ms/epoch - 1ms/step\n",
      "Epoch 5/5\n",
      "540/540 - 1s - loss: 0.1707 - accuracy: 0.9504 - val_loss: 0.1555 - val_accuracy: 0.9567 - 671ms/epoch - 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fbcccf090>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])\n",
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem with change in action function (tanh)\n",
    "\n",
    "Sigmoid the same accuracy but slower training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "540/540 - 1s - loss: 0.3935 - accuracy: 0.8919 - val_loss: 0.1899 - val_accuracy: 0.9440 - 1s/epoch - 3ms/step\n",
      "Epoch 2/5\n",
      "540/540 - 1s - loss: 0.1593 - accuracy: 0.9534 - val_loss: 0.1237 - val_accuracy: 0.9648 - 689ms/epoch - 1ms/step\n",
      "Epoch 3/5\n",
      "540/540 - 1s - loss: 0.1222 - accuracy: 0.9640 - val_loss: 0.1115 - val_accuracy: 0.9668 - 686ms/epoch - 1ms/step\n",
      "Epoch 4/5\n",
      "540/540 - 1s - loss: 0.0995 - accuracy: 0.9701 - val_loss: 0.0908 - val_accuracy: 0.9737 - 686ms/epoch - 1ms/step\n",
      "Epoch 5/5\n",
      "540/540 - 1s - loss: 0.0839 - accuracy: 0.9749 - val_loss: 0.0776 - val_accuracy: 0.9772 - 682ms/epoch - 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fbf060e90>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='tanh'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])\n",
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem with change in BATCH_SIZE (10000)\n",
    "Accuracy is vely low at the first epoch, and the training is slower than the original BATCH_SIZE of 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "6/6 - 1s - loss: 2.1906 - accuracy: 0.2547 - val_loss: 1.8985 - val_accuracy: 0.4823 - 1s/epoch - 174ms/step\n",
      "Epoch 2/5\n",
      "6/6 - 0s - loss: 1.7512 - accuracy: 0.5519 - val_loss: 1.5106 - val_accuracy: 0.6478 - 481ms/epoch - 80ms/step\n",
      "Epoch 3/5\n",
      "6/6 - 1s - loss: 1.3872 - accuracy: 0.6829 - val_loss: 1.2009 - val_accuracy: 0.7210 - 504ms/epoch - 84ms/step\n",
      "Epoch 4/5\n",
      "6/6 - 0s - loss: 1.1075 - accuracy: 0.7425 - val_loss: 0.9676 - val_accuracy: 0.7718 - 485ms/epoch - 81ms/step\n",
      "Epoch 5/5\n",
      "6/6 - 0s - loss: 0.8975 - accuracy: 0.7908 - val_loss: 0.7963 - val_accuracy: 0.8080 - 490ms/epoch - 82ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fbf2038d0>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "BUFFER_SIZE = 10000 #Use at most 10000 samples at a time\n",
    "\n",
    "shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)\n",
    "\n",
    "validation_data = shuffled_train_and_validation_data.take(num_validation_samples)\n",
    "train_data = shuffled_train_and_validation_data.skip(num_validation_samples)\n",
    "\n",
    "BATCH_SIZE = 10000\n",
    "\n",
    "train_data = train_data.batch(BATCH_SIZE)\n",
    "validation_data = validation_data.batch(num_validation_samples)\n",
    "test_data = scaled_test_data.batch(num_test_samples)\n",
    "\n",
    "validation_inputs, validation_targets = next(iter(validation_data))\n",
    "\n",
    "\n",
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])\n",
    "model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem with change in learning rate (0.0001)\n",
    "Slower disminution of loss and increasing of accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "540/540 - 1s - loss: 1.2250 - accuracy: 0.6792 - val_loss: 0.5726 - val_accuracy: 0.8585 - 1s/epoch - 2ms/step\n",
      "Epoch 2/5\n",
      "540/540 - 1s - loss: 0.4512 - accuracy: 0.8829 - val_loss: 0.3718 - val_accuracy: 0.8985 - 665ms/epoch - 1ms/step\n",
      "Epoch 3/5\n",
      "540/540 - 1s - loss: 0.3426 - accuracy: 0.9061 - val_loss: 0.3052 - val_accuracy: 0.9163 - 668ms/epoch - 1ms/step\n",
      "Epoch 4/5\n",
      "540/540 - 1s - loss: 0.2961 - accuracy: 0.9179 - val_loss: 0.2690 - val_accuracy: 0.9262 - 661ms/epoch - 1ms/step\n",
      "Epoch 5/5\n",
      "540/540 - 1s - loss: 0.2676 - accuracy: 0.9253 - val_loss: 0.2463 - val_accuracy: 0.9313 - 678ms/epoch - 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fcbbfa690>"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "BUFFER_SIZE = 10000 #Use at most 10000 samples at a time\n",
    "\n",
    "shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)\n",
    "\n",
    "validation_data = shuffled_train_and_validation_data.take(num_validation_samples)\n",
    "train_data = shuffled_train_and_validation_data.skip(num_validation_samples)\n",
    "\n",
    "BATCH_SIZE = 100\n",
    "\n",
    "train_data = train_data.batch(BATCH_SIZE)\n",
    "validation_data = validation_data.batch(num_validation_samples)\n",
    "test_data = scaled_test_data.batch(num_test_samples)\n",
    "\n",
    "validation_inputs, validation_targets = next(iter(validation_data))\n",
    "\n",
    "\n",
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])\n",
    "\n",
    "custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)\n",
    "model.compile(optimizer = custom_optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 5\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Same problem with change in learning rate (0.02)\n",
    "Oscilatory result for accuracy at the final stages of the training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "540/540 - 1s - loss: 0.3012 - accuracy: 0.9098 - val_loss: 0.2113 - val_accuracy: 0.9380 - 1s/epoch - 2ms/step\n",
      "Epoch 2/50\n",
      "540/540 - 1s - loss: 0.1938 - accuracy: 0.9448 - val_loss: 0.1987 - val_accuracy: 0.9430 - 674ms/epoch - 1ms/step\n",
      "Epoch 3/50\n",
      "540/540 - 1s - loss: 0.1792 - accuracy: 0.9495 - val_loss: 0.2154 - val_accuracy: 0.9458 - 669ms/epoch - 1ms/step\n",
      "Epoch 4/50\n",
      "540/540 - 1s - loss: 0.1612 - accuracy: 0.9555 - val_loss: 0.1626 - val_accuracy: 0.9575 - 667ms/epoch - 1ms/step\n",
      "Epoch 5/50\n",
      "540/540 - 1s - loss: 0.1642 - accuracy: 0.9564 - val_loss: 0.1402 - val_accuracy: 0.9652 - 661ms/epoch - 1ms/step\n",
      "Epoch 6/50\n",
      "540/540 - 1s - loss: 0.1463 - accuracy: 0.9600 - val_loss: 0.1649 - val_accuracy: 0.9605 - 669ms/epoch - 1ms/step\n",
      "Epoch 7/50\n",
      "540/540 - 1s - loss: 0.1514 - accuracy: 0.9611 - val_loss: 0.1584 - val_accuracy: 0.9595 - 662ms/epoch - 1ms/step\n",
      "Epoch 8/50\n",
      "540/540 - 1s - loss: 0.1414 - accuracy: 0.9629 - val_loss: 0.1620 - val_accuracy: 0.9602 - 664ms/epoch - 1ms/step\n",
      "Epoch 9/50\n",
      "540/540 - 1s - loss: 0.1340 - accuracy: 0.9652 - val_loss: 0.1516 - val_accuracy: 0.9612 - 663ms/epoch - 1ms/step\n",
      "Epoch 10/50\n",
      "540/540 - 1s - loss: 0.1403 - accuracy: 0.9644 - val_loss: 0.1482 - val_accuracy: 0.9617 - 663ms/epoch - 1ms/step\n",
      "Epoch 11/50\n",
      "540/540 - 1s - loss: 0.1321 - accuracy: 0.9668 - val_loss: 0.1573 - val_accuracy: 0.9602 - 664ms/epoch - 1ms/step\n",
      "Epoch 12/50\n",
      "540/540 - 1s - loss: 0.1238 - accuracy: 0.9688 - val_loss: 0.1570 - val_accuracy: 0.9587 - 666ms/epoch - 1ms/step\n",
      "Epoch 13/50\n",
      "540/540 - 1s - loss: 0.1240 - accuracy: 0.9683 - val_loss: 0.1300 - val_accuracy: 0.9677 - 661ms/epoch - 1ms/step\n",
      "Epoch 14/50\n",
      "540/540 - 1s - loss: 0.1306 - accuracy: 0.9678 - val_loss: 0.1423 - val_accuracy: 0.9662 - 669ms/epoch - 1ms/step\n",
      "Epoch 15/50\n",
      "540/540 - 1s - loss: 0.1225 - accuracy: 0.9706 - val_loss: 0.1244 - val_accuracy: 0.9703 - 668ms/epoch - 1ms/step\n",
      "Epoch 16/50\n",
      "540/540 - 1s - loss: 0.1154 - accuracy: 0.9718 - val_loss: 0.1475 - val_accuracy: 0.9615 - 665ms/epoch - 1ms/step\n",
      "Epoch 17/50\n",
      "540/540 - 1s - loss: 0.1190 - accuracy: 0.9709 - val_loss: 0.1461 - val_accuracy: 0.9663 - 661ms/epoch - 1ms/step\n",
      "Epoch 18/50\n",
      "540/540 - 1s - loss: 0.1158 - accuracy: 0.9723 - val_loss: 0.1614 - val_accuracy: 0.9645 - 669ms/epoch - 1ms/step\n",
      "Epoch 19/50\n",
      "540/540 - 1s - loss: 0.1260 - accuracy: 0.9694 - val_loss: 0.1448 - val_accuracy: 0.9660 - 662ms/epoch - 1ms/step\n",
      "Epoch 20/50\n",
      "540/540 - 1s - loss: 0.1230 - accuracy: 0.9708 - val_loss: 0.1262 - val_accuracy: 0.9690 - 662ms/epoch - 1ms/step\n",
      "Epoch 21/50\n",
      "540/540 - 1s - loss: 0.1115 - accuracy: 0.9741 - val_loss: 0.1346 - val_accuracy: 0.9682 - 664ms/epoch - 1ms/step\n",
      "Epoch 22/50\n",
      "540/540 - 1s - loss: 0.1121 - accuracy: 0.9729 - val_loss: 0.1143 - val_accuracy: 0.9727 - 661ms/epoch - 1ms/step\n",
      "Epoch 23/50\n",
      "540/540 - 1s - loss: 0.1059 - accuracy: 0.9741 - val_loss: 0.1173 - val_accuracy: 0.9710 - 661ms/epoch - 1ms/step\n",
      "Epoch 24/50\n",
      "540/540 - 1s - loss: 0.1195 - accuracy: 0.9723 - val_loss: 0.1536 - val_accuracy: 0.9632 - 674ms/epoch - 1ms/step\n",
      "Epoch 25/50\n",
      "540/540 - 1s - loss: 0.1065 - accuracy: 0.9744 - val_loss: 0.1220 - val_accuracy: 0.9705 - 670ms/epoch - 1ms/step\n",
      "Epoch 26/50\n",
      "540/540 - 1s - loss: 0.1055 - accuracy: 0.9746 - val_loss: 0.1323 - val_accuracy: 0.9715 - 663ms/epoch - 1ms/step\n",
      "Epoch 27/50\n",
      "540/540 - 1s - loss: 0.1082 - accuracy: 0.9747 - val_loss: 0.1261 - val_accuracy: 0.9692 - 665ms/epoch - 1ms/step\n",
      "Epoch 28/50\n",
      "540/540 - 1s - loss: 0.0962 - accuracy: 0.9775 - val_loss: 0.1368 - val_accuracy: 0.9722 - 665ms/epoch - 1ms/step\n",
      "Epoch 29/50\n",
      "540/540 - 1s - loss: 0.1059 - accuracy: 0.9749 - val_loss: 0.1416 - val_accuracy: 0.9680 - 661ms/epoch - 1ms/step\n",
      "Epoch 30/50\n",
      "540/540 - 1s - loss: 0.1052 - accuracy: 0.9760 - val_loss: 0.1739 - val_accuracy: 0.9657 - 665ms/epoch - 1ms/step\n",
      "Epoch 31/50\n",
      "540/540 - 1s - loss: 0.1044 - accuracy: 0.9746 - val_loss: 0.1227 - val_accuracy: 0.9728 - 664ms/epoch - 1ms/step\n",
      "Epoch 32/50\n",
      "540/540 - 1s - loss: 0.1086 - accuracy: 0.9755 - val_loss: 0.1337 - val_accuracy: 0.9723 - 662ms/epoch - 1ms/step\n",
      "Epoch 33/50\n",
      "540/540 - 1s - loss: 0.1029 - accuracy: 0.9759 - val_loss: 0.1914 - val_accuracy: 0.9642 - 668ms/epoch - 1ms/step\n",
      "Epoch 34/50\n",
      "540/540 - 1s - loss: 0.1020 - accuracy: 0.9761 - val_loss: 0.1149 - val_accuracy: 0.9737 - 666ms/epoch - 1ms/step\n",
      "Epoch 35/50\n",
      "540/540 - 1s - loss: 0.1028 - accuracy: 0.9757 - val_loss: 0.1786 - val_accuracy: 0.9618 - 664ms/epoch - 1ms/step\n",
      "Epoch 36/50\n",
      "540/540 - 1s - loss: 0.1097 - accuracy: 0.9746 - val_loss: 0.1792 - val_accuracy: 0.9628 - 664ms/epoch - 1ms/step\n",
      "Epoch 37/50\n",
      "540/540 - 1s - loss: 0.1175 - accuracy: 0.9748 - val_loss: 0.1421 - val_accuracy: 0.9715 - 662ms/epoch - 1ms/step\n",
      "Epoch 38/50\n",
      "540/540 - 1s - loss: 0.0955 - accuracy: 0.9770 - val_loss: 0.1095 - val_accuracy: 0.9760 - 665ms/epoch - 1ms/step\n",
      "Epoch 39/50\n",
      "540/540 - 1s - loss: 0.1153 - accuracy: 0.9738 - val_loss: 0.1906 - val_accuracy: 0.9612 - 666ms/epoch - 1ms/step\n",
      "Epoch 40/50\n",
      "540/540 - 1s - loss: 0.1062 - accuracy: 0.9752 - val_loss: 0.1014 - val_accuracy: 0.9760 - 666ms/epoch - 1ms/step\n",
      "Epoch 41/50\n",
      "540/540 - 1s - loss: 0.1067 - accuracy: 0.9761 - val_loss: 0.1377 - val_accuracy: 0.9695 - 664ms/epoch - 1ms/step\n",
      "Epoch 42/50\n",
      "540/540 - 1s - loss: 0.1059 - accuracy: 0.9761 - val_loss: 0.1240 - val_accuracy: 0.9735 - 674ms/epoch - 1ms/step\n",
      "Epoch 43/50\n",
      "540/540 - 1s - loss: 0.0958 - accuracy: 0.9774 - val_loss: 0.1454 - val_accuracy: 0.9715 - 663ms/epoch - 1ms/step\n",
      "Epoch 44/50\n",
      "540/540 - 1s - loss: 0.1001 - accuracy: 0.9770 - val_loss: 0.1409 - val_accuracy: 0.9700 - 664ms/epoch - 1ms/step\n",
      "Epoch 45/50\n",
      "540/540 - 1s - loss: 0.1072 - accuracy: 0.9749 - val_loss: 0.1351 - val_accuracy: 0.9728 - 660ms/epoch - 1ms/step\n",
      "Epoch 46/50\n",
      "540/540 - 1s - loss: 0.1034 - accuracy: 0.9760 - val_loss: 0.1360 - val_accuracy: 0.9667 - 657ms/epoch - 1ms/step\n",
      "Epoch 47/50\n",
      "540/540 - 1s - loss: 0.1070 - accuracy: 0.9764 - val_loss: 0.1747 - val_accuracy: 0.9678 - 658ms/epoch - 1ms/step\n",
      "Epoch 48/50\n",
      "540/540 - 1s - loss: 0.1029 - accuracy: 0.9765 - val_loss: 0.1134 - val_accuracy: 0.9743 - 662ms/epoch - 1ms/step\n",
      "Epoch 49/50\n",
      "540/540 - 1s - loss: 0.0974 - accuracy: 0.9768 - val_loss: 0.1197 - val_accuracy: 0.9732 - 659ms/epoch - 1ms/step\n",
      "Epoch 50/50\n",
      "540/540 - 1s - loss: 0.1070 - accuracy: 0.9775 - val_loss: 0.1086 - val_accuracy: 0.9740 - 660ms/epoch - 1ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.History at 0x27fd2d3e690>"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "BUFFER_SIZE = 10000 #Use at most 10000 samples at a time\n",
    "\n",
    "shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)\n",
    "\n",
    "validation_data = shuffled_train_and_validation_data.take(num_validation_samples)\n",
    "train_data = shuffled_train_and_validation_data.skip(num_validation_samples)\n",
    "\n",
    "BATCH_SIZE = 100\n",
    "\n",
    "train_data = train_data.batch(BATCH_SIZE)\n",
    "validation_data = validation_data.batch(num_validation_samples)\n",
    "test_data = scaled_test_data.batch(num_test_samples)\n",
    "\n",
    "validation_inputs, validation_targets = next(iter(validation_data))\n",
    "\n",
    "\n",
    "input_size = 784\n",
    "output_size = 10\n",
    "hidden_layer_size = 50\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "                                tf.keras.layers.Flatten(input_shape = (28,28,1)),\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Input layer tp hidden layer 1\n",
    "                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),    #Hidden layer 1 to hidden layer 2\n",
    "                                tf.keras.layers.Dense(output_size, activation='softmax'),    #Hidden layer 2 to Output layer \n",
    "                            ])\n",
    "\n",
    "custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)\n",
    "model.compile(optimizer = custom_optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])\n",
    "\n",
    "NUM_EPOCHS = 50\n",
    "\n",
    "model.fit(  train_data, \n",
    "            epochs = NUM_EPOCHS, \n",
    "            validation_data = (validation_inputs, validation_targets), \n",
    "            verbose = 2 \n",
    "            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkMAAAHJCAYAAACG+j24AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAB3KUlEQVR4nO3dd3hUVf7H8ffU9E4aIZRQBUFAQLqIiouyFnTdVYq6IroKuGLfRVhXBBSEBRQ79t4brujP7ipVEUV6SyAhkJCE1Gn398eQgRhKEiZMkvm8nicP4c6dO2dOgPlwzveeYzIMw0BEREQkSJkD3QARERGRQFIYEhERkaCmMCQiIiJBTWFIREREgprCkIiIiAQ1hSEREREJagpDIiIiEtQUhkRERCSoKQyJiIhIUFMYEpEGIysri44dO/L222/X63NERA6nMCQiIiJBTWFIREREgprCkIgc1dChQ3n44YeZMWMGZ5xxBj169ODWW2+lpKSEJ554gsGDB3P66aczceJE9u/f73ue2+3mpZde4o9//CPdunVjyJAhzJkzh4qKiirXX7p0KRdeeCHdunXjkksuYf369dXaUFBQwNSpU+nfvz9du3bl8ssv5/vvv6/V+3C73TzxxBOMGDGCbt260b17d/7yl7/www8/VDnvp59+4q9//Ss9e/akb9++TJ48mT179vgez83N5c4776Rfv3706NGD0aNH8+OPPwJHn6676667GDp0qO/3Y8aM4bbbbmPSpEl0796da665xvf8O+64g4EDB9KlSxf69evHHXfcUaVfDcPg2WefZfjw4XTr1o1zzz2Xp59+GsMw+PLLL+nYsSPffvttlddfuXIlHTt2ZNWqVbXqM5FgYg10A0SkYVu8eDEDBgxg3rx5/PLLLzz00EP8+uuvJCUlcd9995GVlcX9999Ps2bNmDZtGgBTp07lvffe47rrrqNXr16sW7eORx55hN9++42nnnoKk8nE559/zqRJk/jjH//I7bffzm+//cbtt99e5bUrKiq46qqr2LdvH7fccgtJSUm89dZbjBs3jqeeeop+/frV6D3MmTOHV155hVtvvZWOHTuyZ88eHnnkEW6++Wa+/PJLwsLCWLduHaNHj+a0007jwQcfxO1289BDD3Httdfy7rvvUlFRwRVXXIHb7eb2228nOTmZxYsX89e//pV33nkHq7Xm/5x+/PHHXHjhhTz66KN4PB7KysoYO3YscXFxTJs2jaioKH788UcefvhhQkND+fe//w3Agw8+yHPPPcc111zDgAEDWLt2LXPmzMHlcjFu3DiSkpJ47733GDhwoO+13n33XVq3bs3pp59e4/aJBBuFIRE5psjISObNm4fVaqV///6888477NmzhzfeeIOoqCgAvvnmG1avXg3A5s2befPNN7n11lsZP348AAMGDCApKYk77riDr7/+mjPPPJNHHnmEbt26MXv2bAAGDRoEwEMPPeR77ffee4/169fz+uuvc9pppwEwePBgxowZw5w5c3jrrbdq9B5yc3O55ZZbGDNmjO9YSEgIEydOZMOGDXTv3p3HHnuM2NhYFi9eTEhICABJSUnceuutbNq0iVWrVrFr1y7eeecdTjnlFAB69uzJxRdfzIoVK2oczABsNhv33nsvdrsdgN9++42UlBQeeOAB0tPTAejbty9r1qxh+fLlABQVFfH8888zevRoX2js378/e/fuZcWKFVx//fVccsklvPDCC5SUlBAREUF5eTkff/yx7+cgIkemaTIROaZu3bpVGfVo1qwZbdq08QUhgNjYWA4cOADg+/C+4IILqlznggsuwGKxsGzZMsrLy/n1118566yzqpwzfPjwKr///vvvSUxMpEuXLrhcLlwuF263m7POOotffvmFwsLCGr2Hhx56iKuuuor8/HxWrlzJW2+9xfvvvw+Aw+EAYNWqVQwePNgXhAB69OjB559/zimnnMKqVato0aKFLwgBhIWF8cknn/CnP/2pRu2olJGR4QtCAKeccgovv/wyaWlpbN++na+++oqnn36arVu3+tr3008/4XK5GDZsWJVrTZkyhaeeegqASy+9lNLSUj799FMAPv30U0pLS7n44otr1T6RYKORIRE5psjIyGrHwsPDj3p+ZUBJTEysctxqtRIXF8eBAwcoLCzEMAzi4uKqnJOUlFTl9wUFBezdu5cuXboc8bX27t1LaGjocd/D2rVruffee1m7di1hYWG0a9eO5s2bA946nMrXSkhIOOo1jvd4bURERFQ79swzz/DYY49RUFBAs2bNOPXUUwkLC/OFzIKCAgDi4+OPet1WrVrRp08f3n33XS6++GLeffdd+vfvT3Jysl/aLdJUKQyJiF/FxMQA3qCSlpbmO+50Otm/fz9xcXHExsZiNpvZt29fledWfuBXioqKonXr1syZM+eIr9WiRYtq1/i94uJixo0bR8eOHfnoo4/IyMjAbDbz1Vdf8cknn1R5rfz8/GrP/+qrrzjllFOIiooiKyur2uOrV68mJibGF8rcbneVx0tLS4/ZPoAPPviAWbNmcfvttzNy5Ehf4Ln55ptZu3YtANHR0QDk5+eTkZHhe+7u3bvZuXMnp59+OjabjUsvvZR//OMfbNmyhe+///6ofScih2iaTET8qk+fPgB89NFHVY5/9NFHuN1uTj/9dEJCQujRowdLly71jcwAfP7559WulZ2dTUJCAl27dvV9fffddzz11FNYLJbjtmfr1q0UFBQwduxY2rVrh9ns/Wfv66+/BsDj8QDQq1cvvvvuO9+0FMC6desYP348v/76K7169SIzM5NNmzb5Hq+oqGDixIm8+eabvhG0w+8+czqd/Pzzz8dt46pVq4iOjmbcuHG+IFRSUsKqVat87evWrRs2m40vvviiynMXL17M5MmTfX1x3nnnERYWxr/+9S8iIiI455xzjvv6IsFOI0Mi4lft2rXjkksuYcGCBZSVldG7d29+++03Hn74Yc444wxfofTkyZO56qqrmDBhAn/+85/Ztm0bjz32WJVrjRw5khdffJFrrrmGG264gdTUVP73v//x5JNPMnr0aGw223Hb06ZNGyIjI3nsscewWq1YrVY++eQT3nzzTQDKysoAuPHGG/nzn//M9ddfz9ixYykvL+c///kP3bp1Y8CAATgcDl544QX+9re/MWnSJOLi4nj++edxOp1ceeWVxMTE0KNHD1544QVatWpFTEwMzz//POXl5cecVgRv0HnllVeYNWsWZ511Frm5uTz99NPs27fPN9IWHx/P2LFjefbZZ7Hb7fTp04c1a9bwyiuvcMcdd/hCXlhYGBdccAGvvfYaV1xxRZXaJBE5Mo0MiYjf3X///dx000188MEHjB8/npdeeomxY8fy5JNP+j60e/XqxZNPPsmePXuYMGECr732GjNmzKhynfDwcF566SVOP/10Zs+ezXXXXcfSpUu59dZbufvuu2vUlqioKBYtWoRhGNx8883ccccd7N69mxdffJGIiAhWrlwJQOfOnXnhhRdwuVz8/e9/Z/r06Zx++uk8/vjj2O12IiMjefHFFznttNO47777+Pvf/47H4+H555/33QE2a9YsTj31VKZMmcLd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CA90c0RERBo9haFGZufBeqHmCRFYtdeMiIjICdOnaSOTWVkvpOJpERERv1AYamQOrTyteiERERF/UBhqZCrDUEvdSSYiIuIXCkONSGm5k32F5YCmyURERPxFYagRqRwVSogOISLUFuDWiIiINA0KQ42I6oVERET8T2GoEdl5MAy1UL2QiIiI3ygMNSIqnhYREfE/haFGwu3xsGtvCaDiaREREX9SGGokcvJKcbk9hNgtJMaGBbo5IiIiTYbCUCNRWS+UnhiJ2WQKcGtERESaDoWhRsJ3J5mmyERERPxKYaiRyNxzcE8yFU+LiIj4lcJQI3FojSGFIREREX9SGGoECosrKCp1YjJBi0SFIREREX9SGGoEKounk+PCCbFZAtwaERGRpkVhqBHwLbao4mkRERG/UxhqBHaqeFpERKTeKAw1AtqgVUREpP4oDDVwDqebnPxSQCNDIiIi9UFhqIHbta8Ew4CocBuxkfZAN0dERKTJURhq4A5fX8ikbThERET8TmGogVPxtIiISP1SGGrgfLfVq3haRESkXigMNWAew9A2HCIiIvVMYagB21dYTrnDjdViIiUhPNDNERERaZIUhhqwyp3qmzeLwGrRj0pERKQ+6BO2AVO9kIiISP1TGGrAVC8kIiJS/xSGGrCdexSGRERE6pvCUANVWu4kr6gcgHTtVi8iIlJvFIYaqMopsoToECJCbQFujYiISNOlMNRA7dRO9SIiIieFwlADlal6IRERkZNCYaiB8t1Wr3ohERGReqUw1AC53B527dPIkIiIyMmgMNQA5eSX4nIbhNgtNIsNC3RzREREmjSFoQbo8MUWzSZTgFsjIiLStAU8DHk8HhYsWMCgQYPo3r071113HZmZmUc9f/v27YwfP55evXoxePBgFixYgMvlOuK5+fn5DBw4kIULF9ZX8+uFiqdFREROnoCHoUWLFvHyyy9z33338eqrr+LxeBg3bhwOh6PauYWFhYwaNYqysjKee+455s6dy8cff8zUqVOPeO0pU6awd+/e+n4LfpeZ692gtaXCkIiISL0LaBhyOBwsXryYSZMmMWTIEDp16sS8efPIyclh6dKl1c5/5513KC0tZf78+XTp0oVevXoxffp03nrrLbKysqqc+9prr7F9+3YSExNP1tvxC8MwtMaQiIjISRTQMLR+/XpKSkro16+f71h0dDSdO3dmxYoV1c7fsWMHGRkZxMfH+4517twZgJUrV/qObdu2jTlz5jB79mzsdns9vgP/KyxxcKDUickEaYkRgW6OiIhIk2cN5Ivn5OQAkJqaWuV4UlKS77HfH8/NzcXtdmOxWADYtWsXAHl5eQA4nU5uvfVWrr32Wrp06eLX9lqt/s2OFou5yq8Au/aVAJASH05EmLbhqIkj9aPUnvrRP9SP/qF+9A/1Y80ENAyVlZUBVBu9CQkJobCwsNr5w4cPZ9GiRcycOZPJkydTWlrK9OnTsVqtOJ1OABYsWEBISAjXXXedX9tqNpuIi6ufkZro6EO3z+8t8oa79ulx9fZ6TdXh/Sh1p370D/Wjf6gf/UP9eGwBDUOhoaGAt3ao8nuAiooKwsKq/+Bat27N/PnzmTp1Ki+99BLh4eFMnDiRzZs3ExUVxfLly3nllVd45513fCNH/uLxGBQVlfr1mhaLmejoMIqKynC7PQBs3J4PQHJcGPv3l/j19ZqqI/Wj1J760T/Uj/6hfvSPYO/H6OiwGo2KBTQMVU6P5ebm0rJlS9/x3NxcOnbseMTnDB06lKFDh5Kbm0tsbCwul4tZs2aRnp7uK7C+8MILfeeXlZXx+OOP89///pePPvrohNrrctXPHyS32+O79o493jvJ0ppF1NvrNVWH96PUnfrRP9SP/qF+9A/147EFNAx16tSJyMhIli1b5gtDRUVFrFu3jtGjR1c7f+XKlcyfP59nnnmGpKQkAJYsWUJYWBg9e/akS5cu3HDDDVWeM2bMGIYNG8Y111xT/2/oBFU43eTke0eftCeZiIjIyRHQMGS32xk9ejRz5swhPj6etLQ0Zs+eTUpKCsOGDcPtdpOfn09UVBShoaFkZGSwYcMGHnjgAcaOHcuGDRuYPn06119/PZGRkURGRpKQkFDlNaxWKzExMaSlpQXoXdbcrr0lGAZEhduIiWhcd8GJiIg0VgENQwCTJk3C5XIxZcoUysvL6d27N08//TQ2m42srCzOPvtsZs6cyciRI4mPj+exxx5j1qxZjBgxgsTERCZMmMDVV18d6LfhF4cvtmjSNhwiIiInhckwDCPQjWgM3G4P+fn+LWi2Ws3ExUWwf38JLpeHF5Zu4IvVu/hDn5ZcPrSdX1+rKft9P0rdqB/9Q/3oH+pH/wj2foyPj6hRAbUWHmhAfBu0ql5IRETkpFEYaiA8hlFlt3oRERE5ORSGGoh9BWVUONxYLSZS4sMD3RwREZGgoTDUQFSOCqU1i8SqZdNFREROGn3qNhA792iKTEREJBAUhhoIFU+LiIgEhsJQA3H4GkMiIiJy8igMNQAlZU7yiioATZOJiIicbApDDcDOg5uzJkSHEh5qC3BrREREgovCUAOwU+sLiYiIBIzCUANQOTKknepFREROPoWhBkC31YuIiASOwlCAudwedu2tvK0+KsCtERERCT4KQwGWlVuMy20QarfQLCY00M0REREJOgpDAbZtdyHgnSIzm0wBbo2IiEjwURgKsK27DoUhEREROfkUhgJs++4iAFqqXkhERCQgFIYCyDAMtmVrZEhERCSQFIYCqKDYQWGxA5MJ0ppFBLo5IiIiQUlhKIAqF1tMTYjAbrMEuDUiIiLBSWEogHwrT2uKTEREJGAUhgKocuXplikqnhYREQkUhaEA0p5kIiIigacwFEDFZU4sZhOtdFu9iIhIwFgD3YBgNumyboSE2omJDMHl8gS6OSIiIkFJYSiAOraMIy4ugv37SwLdFBERkaClaTIREREJagpDIiIiEtQUhkRERCSoKQyJiIhIUFMYEhERkaCmMCQiIiJBTWFIREREgprCkIiIiAQ1hSEREREJagpDIiIiEtQUhkRERCSoKQyJiIhIUFMYEhERkaBmMgzDCHQjGgPDMPB4/N9VFosZt9vj9+sGG/Wjf6gf/UP96B/qR/8I5n40m02YTKbjnqcwJCIiIkFN02QiIiIS1BSGREREJKgpDImIiEhQUxgSERGRoKYwJCIiIkFNYUhERESCmsKQiIiIBDWFIREREQlqCkMiIiIS1BSGREREJKgpDImIiEhQUxgSERGRoKYwJCIiIkFNYSgAPB4PCxYsYNCgQXTv3p3rrruOzMzMQDer0SkoKGDq1KkMHjyYnj17csUVV7By5cpAN6tR27ZtGz169ODtt98OdFMapXfffZfzzz+frl27csEFF/Dxxx8HukmNjsvlYv78+Zx11ln06NGDUaNG8dNPPwW6WY3K448/zpgxY6oc++233xg9ejTdu3dn6NChPP/88wFqXcOkMBQAixYt4uWXX+a+++7j1VdfxePxMG7cOBwOR6Cb1qhMnjyZH3/8kblz5/LWW29xyimncO2117J169ZAN61Rcjqd3HbbbZSWlga6KY3Se++9xz//+U9GjRrFRx99xIgRI3x/RqXmHn30Ud544w3uu+8+3n33Xdq0acO4cePIzc0NdNMahZdeeon//Oc/VY7t37+fa665hpYtW/LWW29x0003MWfOHN56663ANLIBUhg6yRwOB4sXL2bSpEkMGTKETp06MW/ePHJycli6dGmgm9do7Nixg++++45//etf9OrVizZt2nDPPfeQlJTEBx98EOjmNUoLFy4kMjIy0M1olAzDYP78+YwdO5ZRo0bRsmVL/va3v9G/f3+WL18e6OY1Kp999hkjRoxg4MCBtGrVirvuuosDBw5odOg49uzZww033MCcOXNo3bp1lcdef/11bDYb//73v2nbti2XXnopV199NU888URgGtsAKQydZOvXr6ekpIR+/fr5jkVHR9O5c2dWrFgRwJY1LnFxcTzxxBN07drVd8xkMmEymSgqKgpgyxqnFStW8NprrzFr1qxAN6VR2rZtG7t27eKPf/xjleNPP/00119/fYBa1TglJCTwxRdfkJWVhdvt5rXXXsNut9OpU6dAN61B+/XXX7HZbLz//vucdtppVR5buXIlffr0wWq1+o717duX7du3s2/fvpPd1AZJYegky8nJASA1NbXK8aSkJN9jcnzR0dGceeaZ2O1237FPPvmEHTt2MGjQoAC2rPEpKirijjvuYMqUKdX+XErNbNu2DYDS0lKuvfZa+vXrx5/+9Cc+//zzALes8fnnP/+JzWbj7LPPpmvXrsybN48FCxbQsmXLQDetQRs6dCgLFy4kPT292mM5OTmkpKRUOZaUlARAdnb2SWlfQ6cwdJKVlZUBVPkQBwgJCaGioiIQTWoSVq9ezd13382wYcMYMmRIoJvTqPzrX/+iR48e1UY1pOaKi4sBuPPOOxkxYgSLFy9mwIAB3HjjjXz//fcBbl3jsnnzZqKionjkkUd47bXXGDlyJLfddhu//fZboJvWaJWXlx/xMwfQ585B1uOfIv4UGhoKeGuHKr8H7x/IsLCwQDWrUfvss8+47bbb6NmzJ3PmzAl0cxqVd999l5UrV6rO6gTZbDYArr32Wi655BIATjnlFNatW8czzzxTZVpcji47O5tbb72VZ599ll69egHQtWtXNm/ezMKFC1m0aFGAW9g4hYaGVrtBpzIEhYeHB6JJDY5Ghk6yymmI398ZkZubS3JyciCa1Ki9+OKLTJw4kbPOOovHHnvM978dqZm33nqLvLw8hgwZQo8ePejRowcA06ZNY9y4cQFuXeNR+Xe3Q4cOVY63a9eOrKysQDSpUVqzZg1Op7NKLSDAaaedxo4dOwLUqsYvJSXliJ85gD53DtLI0EnWqVMnIiMjWbZsmW8OvKioiHXr1jF69OgAt65xqVyeYMyYMfzzn//EZDIFukmNzpw5cygvL69ybNiwYUyaNIkLL7wwQK1qfLp06UJERARr1qzxjWgAbNy4UbUutVBZ17Jhwwa6devmO75x48Zqd0hJzfXu3ZtXX30Vt9uNxWIB4IcffqBNmzYkJCQEuHUNg8LQSWa32xk9ejRz5swhPj6etLQ0Zs+eTUpKCsOGDQt08xqNbdu2MWPGDM4991yuv/76KndEhIaGEhUVFcDWNR5H+19hQkKC/sdYC6GhoYwbN45HHnmE5ORkunXrxkcffcR3333Hs88+G+jmNRrdunXj9NNP584772TatGmkpKTw7rvv8v333/PKK68EunmN1qWXXspTTz3FP//5T8aNG8fPP//Ms88+y7333hvopjUYCkMBMGnSJFwuF1OmTKG8vJzevXvz9NNP++oO5Pg++eQTnE4nn376KZ9++mmVxy655BLdIi4n3Y033khYWBjz5s1jz549tG3bloULF3LGGWcEummNhtls5tFHH+U///kPd999N4WFhXTo0IFnn3222u3iUnMJCQk89dRT3H///VxyySUkJiZyxx13+OrbBEyGYRiBboSIiIhIoKiAWkRERIKawpCIiIgENYUhERERCWoKQyIiIhLUFIZEREQkqCkMiYiISFBTGBIREZGgpjAkIiIiQU1hSESkjoYOHcpdd90V6GaIyAlSGBIREZGgpjAkIiIiQU1hSEQanTfeeIMLLriAU089lSFDhrBw4ULcbjcAd911F2PGjOHNN9/krLPOokePHlx11VWsX7++yjW2b9/OpEmTGDBgAN27d2fMmDGsWrWqyjnFxcXcd999DBo0iO7du3PppZfy5ZdfVjnH6XTy4IMP+q7z17/+lR07dtTr+xcR/1IYEpFG5fHHH+eee+6hX79+PPbYY4waNYonn3ySe+65x3fOb7/9xrx585gwYQKzZ89m//79jB49mtzcXAA2b97MyJEjycrKYsqUKcyZMweTycRVV13F8uXLAXC73fz1r3/lgw8+4Prrr2fRokVkZGRw0003sXLlSt9rLVmyhE2bNjFr1iymTZvGL7/8wi233HJyO0VETog10A0QEampAwcOsGjRIv785z8zZcoUAAYOHEhsbCxTpkzhmmuu8Z332GOP0atXLwC6devGOeecw/PPP89tt93Gww8/jN1u5/nnnycyMhKAIUOGMGLECB588EHefPNNvv76a9asWcMjjzzCOeecA0Dfvn3JzMzkhx9+8F07OTmZRYsWYbPZANixYwePPvooxcXFvmuLSMOmMCQijcaPP/5IeXk5Q4cOxeVy+Y4PHToUgO+++w6AFi1a+MIKQFJSEj169GDFihUALF++nLPOOqtKWLFarVxwwQU88sgjlJSUsGrVKmw2m+/aAGazmVdffbVKm7p16+YLQpWvDVBUVKQwJNJIKAyJSKNRUFAAwPjx44/4eOU0WHJycrXHEhIS+PXXXwEoLCykWbNm1c5p1qwZhmFQXFxMQUEBsbGxmM3HriYIDw+v8vvK8z0ez7HfjIg0GApDItJoREdHAzBnzhxat25d7fFmzZoxf/589u/fX+2xffv2kZCQAEBMTAz79u2rds7evXsBiIuLIyoqioKCAgzDwGQy+c5Zt24dhmHQpUsXf7wlEWkAVEAtIo3Gaaedhs1mY8+ePXTt2tX3ZbVamTt3LllZWYD3TrEtW7b4nrdnzx5+/PFH+vXrB0Dv3r354osvKC4u9p3jdrv56KOP6Nq1K3a7nV69euF0Ovn666995xiGwd13383jjz9+kt6xiJwMGhkSkUYjLi6OcePGMX/+fIqLiznjjDPYs2cP8+fPx2Qy0alTJ8AbWm644QZuueUWLBYLDz/8MDExMYwZMwaACRMm8PXXXzN27FjGjx+PzWbjxRdfJDMzk6eeegrwFlT36NGDu+66i7///e+kp6fz3nvvsWXLFu67776A9YGI+J/CkIg0Kn//+99JTEzk5Zdf5qmnniImJoZ+/foxefJkoqKiAGjevDl//etfmTFjBmVlZfTv359HH32U2NhYANq3b8/LL7/M3LlzufvuuzGZTHTr1o3nn3/eV3htsVh48sknmTNnDvPnz6esrIyOHTuyePFiunXrFqi3LyL1wGQYhhHoRoiI+Mtdd93F8uXL+fzzzwPdFBFpJFQzJCIiIkFNYUhERESCmqbJREREJKhpZEhERESCmsKQiIiIBDWFIREREQlqCkMiIiIS1BSGREREJKgpDImIiEhQUxgSERGRoKYwJCIiIkHt/wEF6nh/LBmQFQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkMAAAHJCAYAAACG+j24AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABk8ElEQVR4nO3dd3xV9f3H8de5M3snBAgoIEPZCCICioK4bR217c9VLY5WS+vWSm2tdYLixIXWtmq14qp1gdtaFBFR9h4JEJKQve48vz9uciGGEcIJ997c9/Px4JFw7rnffO6Hm+TN93zPOYZpmiYiIiIiccoW6QJEREREIklhSEREROKawpCIiIjENYUhERERiWsKQyIiIhLXFIZEREQkrikMiYiISFxTGBIREZG4pjAkIiIicU1hSEQ6paKiIvr3789rr71m6XNOOOEEbr75ZitKFJEooTAkIiIicU1hSEREROKawpCIHBQnnHACjz76KHfddRejR49m+PDhXHfdddTV1fHUU09x7LHHcuSRR/Kb3/yGioqK8PMCgQAvvPACZ5xxBkOGDGHChAnMmDEDj8fTYvy5c+dy5plnMmTIEM466yxWrlzZqobKykpuu+02jjnmGAYPHsx5553H/PnzD+h11dTUcPfddzNp0iQGDx7M6aefzpw5c1rss3TpUi6++GKOPPJIhg8fzi9+8QsWL14cfry8vJzrrruOsWPHMnjwYH70ox/xxhtvHFBdItJ2jkgXICLx49lnn2Xs2LHMnDmTpUuXcv/997Ns2TLy8vK44447KCoq4s477yQnJ4c//vGPANx22228+eabXHbZZYwcOZLly5fz2GOPsWLFCmbPno1hGHz00UdMnTqVM844gxtuuIEVK1Zwww03tPjaHo+Hiy++mLKyMq655hry8vJ49dVXmTJlCrNnz2bMmDH7/XoaGxv5v//7P3bs2MHUqVPp3r07H3zwAbfeeitlZWVceeWV1NbWMmXKFI4++mgeeeQRvF4vjz/+OL/85S/55JNPSE1N5YYbbmDHjh3cfvvtpKSk8Oabb3LTTTeRn5/P0UcfbUnvRWTPFIZE5KBJSUlh5syZOBwOjjnmGF5//XW2b9/OK6+8QmpqKgCff/45ixYtAmDt2rXMmTOH6667jssvvxyAsWPHkpeXx4033shnn33Gcccdx2OPPcaQIUOYPn06AOPH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gQNJTk7mu+++C89oAKxevVprXfZD87qWVatWMWTIkPD21atXtzpDStpu1KhRvPTSSwQCAex2OwBffvklvXr1Ijs7O8LVRQeFoYPM5XJxwQUXMGPGDLKysujevTvTp08nPz+fyZMnR7q8mLFhwwbuuusuTjzxRK644ooWZ0QkJCSQmpoawepix57+V5idna3/Me6HhIQEpkyZwmOPPUaXLl0YMmQIb7/9Nl988QXPPfdcpMuLGUOGDOHII4/kpptu4o9//CP5+fm88cYbzJ8/n3/+85+RLi9mnXPOOcyePZtbb72VKVOm8P333/Pcc89x++23R7q0qKEwFAFTp07F7/czbdo0GhsbGTVqFM8880x43YHs2/vvv4/P52PevHnMmzevxWNnnXWWThGXg+7Xv/41iYmJzJw5k+3bt9OnTx8eeeQRRo8eHenSYobNZuPxxx/nwQcf5JZbbqGqqop+/frx3HPPtTpdXNouOzub2bNnc+edd3LWWWeRm5vLjTfeGF7fJmCYpmlGuggRERGRSNECahEREYlrCkMiIiIS1xSGREREJK4pDImIiEhcUxgSERGRuKYwJCIiInFNYUhERETimsKQiIiIxDWFIRGRdjrhhBO4+eabI12GiBwghSERERGJawpDIiIiEtcUhkQk5rzyyiucdtppDBo0iAkTJvDII48QCAQAuPnmm7nwwguZM2cOxx9/PMOHD+fiiy9m5cqVLcbYuHEjU6dOZezYsQwbNowLL7yQb775psU+tbW13HHHHYwfP55hw4Zxzjnn8Mknn7TYx+fzcd9994XHufTSS9m0aVOHvn4RsZbCkIjElCeffJI//OEPjBkzhieeeILzzz+fp59+mj/84Q/hfVasWMHMmTO5+uqrmT59OhUVFVxwwQWUlJQAsHbtWs4++2yKioqYNm0aM2bMwDAMLr74YhYsWABAIBDg0ksv5a233uKKK65g1qxZ9O7dm6uuuoqFCxeGv9Y777zDmjVruOeee/jjH//I0qVLueaaaw5uU0TkgDgiXYCISFvV1NQwa9YsfvrTnzJt2jQAxo0bR0ZGBtOmTeOSSy4J7/fEE08wcuRIAIYMGcKkSZP4+9//zvXXX8+jjz6Ky+Xi73//OykpKQBMmDCB008/nfvuu485c+bw2Wef8d133/HYY48xadIkAI4++mgKCwv58ssvw2N36dKFWbNm4XQ6Adi0aROPP/44tbW14bFFJLopDIlIzPj2229pbGzkhBNOwO/3h7efcMIJAHzxxRcAFBQUhMMKQF5eHsOHD+frr78GYMGCBRx//PEtworD4eC0007jscceo66ujm+++Qan0xkeG8Bms/HSSy+1qGnIkCHhINT8tQGqq6sVhkRihMKQiMSMyspKAC6//PLdPt58GKxLly6tHsvOzmbZsmUAVFVVkZOT02qfnJwcTNOktraWyspKMjIysNn2vpogKSmpxd+b9w8Gg3t/MSISNRSGRCRmpKWlATBjxgwOPfTQVo/n5OTw0EMPUVFR0eqxsrIysrOzAUhPT6esrKzVPqWlpQBkZmaSmppKZWUlpmliGEZ4n+XLl2OaJgMHDrTiJYlIFNACahGJGUOHDsXpdLJ9+3YGDx4c/uNwOHjggQcoKioCQmeKrVu3Lvy87du38+233zJmzBgARo0axccff0xtbW14n0AgwNtvv83gwYNxuVyMHDkSn8/HZ599Ft7HNE1uueUWnnzyyYP0ikXkYNDMkIjEjMzMTKZMmcJDDz1EbW0to0ePZvv27Tz00EMYhsGAAQOAUGi58sorueaaa7Db7Tz66KOkp6dz4YUXAnD11Vfz2WefcdFFF3H55ZfjdDp5/vnnKSwsZPbs2UBoQfXw4cO5+eab+d3vfkePHj148803WbduHXfccUfEeiAi1lMYEpGY8rvf/Y7c3FxefPFFZs+eTXp6OmPGjOHaa68lNTUVgG7dunHppZdy11130dDQwDHHHMPjjz9ORkYGAH379uXFF1/kgQce4JZbbsEwDIYMGcLf//738MJru93O008/zYwZM3jooYdoaGigf//+PPvsswwZMiRSL19EOoBhmqYZ6SJERKxy8803s2DBAj766KNIlyIiMUJrhkRERCSuKQyJiIhIXNNhMhEREYlrmhkSERGRuKYwJCIiInFNYUhERETimsKQiIiIxDWFIREREYlrCkMiIiIS1xSGREREJK4pDImIiEhc+3+79Fp+YiutwwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(history.history.keys())\n",
    "\n",
    "# summarize history for accuracy\n",
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.title('model accuracy')\n",
    "plt.ylabel('accuracy')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'validation'], loc='upper left')\n",
    "plt.show()\n",
    "# summarize history for loss\n",
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'validation'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 82ms/step - loss: 0.0902 - accuracy: 0.9790\n",
      "Test loss: 0.09. Test accuracy: 97.90\n"
     ]
    }
   ],
   "source": [
    "test_loss, test_accuracy = model.evaluate(test_data)\n",
    "\n",
    "print(\"Test loss: {0:.2f}. Test accuracy: {1:.2f}\".format(test_loss, test_accuracy*100.))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "96cfe203e9447c5a24b4e9889d7cd19a7ac0b3379138290f54f8d60661b21395"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
