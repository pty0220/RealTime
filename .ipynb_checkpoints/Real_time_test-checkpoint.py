import os

from keras.layers import Layer, Input, Dropout, Conv3D, Activation, add, BatchNormalization, UpSampling3D, \
    Conv3DTranspose, Flatten, concatenate, MaxPooling3D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Network
import SimpleITK as sitk
import tensorflow as tf
import numpy as np


a=1