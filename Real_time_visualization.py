import os

from keras.layers import Layer
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Network
from keras.models import model_from_json

import SimpleITK as sitk
import tensorflow as tf
import numpy as np


import os
import vtk
from  datetime import datetime
import volume as vlm
import helpfunction as hlp
import cv2
import keras
from make_transducer import makeTransducer_vertical
from vtk.util import numpy_support as ns
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  # Select GPU device±

ARC = np.loadtxt('ARC_result')
trans = ARC[:,4:]
trans[:,0] = -trans[:,0]
trans[:,1] = -trans[:,1]

target = [-22.7292, 27.1715, 56.572]


def rotate(x, y, r):
    rx = (x*math.cos(r)) - (y*math.sin(r))
    ry = (y*math.cos(r)) + (x*math.sin(r))
    return (rx, ry)

# create a ring of points centered on center (x,y) with a given radius
# using the specified number of points
# center should be a tuple or list of coordinates (x,y)
# returns a list of point coordinates in tuples
# ie. [(x1,y1),(x2,y2
def point_ring(center, num_points, radius):
    arc = (2 * math.pi) / num_points # what is the angle between two of the points
    points = []
    for p in range(num_points):
        (px,py) = rotate(0, radius, arc * p)
        if py<0:
            py = -py
        px += center[1]
        py += center[2]
        points.append((px,py))
    return points



def make_analysis_range2(num_pts, radius, range_angle, Target=[0,0,0]):

    # num_pts     : number of transducer (setting value, int)
    # range_angle : analysis range angle (setting value, degree)
    # radius      : transducer focal size

    # calculate height of analysis range
    # Pythagorean theorem (under three lines)
    h_wid = radius*np.sin(np.deg2rad(range_angle))
    p_height = radius ** 2 - h_wid ** 2
    height_from_center = np.sqrt(p_height)
    # height of analysis range
    height = radius - height_from_center

    # ratio height/radius*2
    rate = height / (radius * 2)

    # make evenly distributed sphere
    indices_theta = np.arange(0, num_pts, dtype=float)
    indices_phi = np.linspace(0, num_pts * rate, num=num_pts)  ## define transdcuer's height as ratio

    phi = np.arccos(1 - 2 * indices_phi / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices_theta

    # x,y,z is coordination of evenly distributed sphere
    # multiply radius(focal size)
    x, y, z = np.cos(theta) * np.sin(phi)*radius, np.sin(theta) * np.sin(phi)*radius, np.cos(phi)*radius;

    coordi  = np.zeros((num_pts, 3))
    dis_min = np.zeros((num_pts,1))
    x = x + Target[0]
    y = y + Target[1]
    z = z + Target[2]

    #z[z<Target[2]] = target[2]
    #y = Target[1]

    coordi[:,0] = x
    coordi[:,1] = y
    coordi[:,2] = z

    return coordi



def DICOM_read_vertical(filePath):

    if filePath[-2:] == "gz":
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(filePath)
        image = reader.Execute()
    elif filePath[-3:] == "nii":
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(filePath)
        image = reader.Execute()
    elif filePath[-4:] == "nrrd":
        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(filePath)
        image = reader.Execute()
    else:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(filePath)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

    numpy_array = sitk.GetArrayFromImage(image)

    return numpy_array, image


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        size_increase = [0, 2 * self.padding[0], 2 * self.padding[1], 2 * self.padding[2], 0]
        output_shape = list(s)

        for i in range(len(s)):
            if output_shape[i] == None:
                continue
            output_shape[i] += size_increase[i]

        return tuple(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera, ReflectionPadding3D):

    def __init__(self, interactor, renderer, result_volume, input_data):
        self.AddObserver("KeyPressEvent", self.keyPressEvent)
        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()
        self.iren = interactor
        self.ren = renderer
        self.volume = result_volume
        self.skull = input_data
        self.image = self.volume.GetMapper().GetInput()
        self.target_pos = np.array([-22.7292, 27.1715, 56.572])

        self.dimension = self.image.GetDimensions()
        self.origin = self.image.GetOrigin()
        self.spacing = self.image.GetSpacing()

        x_end = origin[0] + spacing[0] * (dimension[0] - 1)
        y_end = origin[1] + spacing[1] * (dimension[1] - 1)
        z_end = origin[2] + spacing[2] * (dimension[2] - 1)

        x_arr = np.linspace(origin[0], x_end, dimension[0])
        y_arr = np.linspace(origin[1], y_end, dimension[1])
        z_arr = np.linspace(origin[2], z_end, dimension[2])

        self.image_cordi = [x_arr, y_arr, z_arr]

        points = np.array(point_ring(target, 100, 55))
        trans_pos = np.zeros((100, 3))
        trans_pos[:, 0] = target[0]#points[:, 0]
        trans_pos[:, 1] = points[:, 0]
        trans_pos[:, 2] = points[:, 1]

        trans_pos = trans_pos[trans_pos[:, 1].argsort()]

        self.trans_pos = trans_pos

        model_path = "F:\\model\\Medical-Image-Synthesis-multiview-Resnet_patch3D\\runs\\20201208-180043-T1-CT_cropped_bias_P9_LR_0.0002_RL_9_DF_64_GF_32_RF_70\\models\\"

        json_file = open(model_path + "G_A2B_model_epoch_360.json", 'r')
        loded_model_json = json_file.read()
        json_file.close()

        G_A2B = model_from_json(loded_model_json, custom_objects={'ReflectionPadding3D': ReflectionPadding3D})
        G_A2B.load_weights(model_path + 'G_A2B_model_epoch_360.hdf5')

        self.G_A2B = G_A2B
        self.num = 0


        # num= 50
        # gather_input = np.zeros((num,140,140,140))
        # for i in range(num):
        #     input_data, itk_image = DICOM_read_vertical(data_path + "input_image"+str(i)+".nii")
        #     print(i)
        #     gather_input[i] = input_data
        #
        # gather_output = self.G_A2B.predict(gather_input[:,:,:,:,np.newaxis])
        # self.gather_output = np.squeeze(gather_output)

    def update(self):
        self.iren.GetRenderWindow().Render()

    def keyPressEvent(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'Up':
            s = datetime.now()
            self.num = self.num+1
            self.live_pressure(self.num)
            self.update()
            e = datetime.now()
            print("total function update: ", e-s)

        if key == 'Down':
            self.num = self.num-1
            self.live_pressure(self.num)
            self.update()

        if key == 'Left':
            self.update()

        if key == 'Right':
            self.num = self.num+1
            self.live_pressure(self.num)
            self.update()

        return


    def live_pressure(self, num):
        # data_path = "F:\\syntheticGANs\\data\\T1-CT_cropped_bias_P9\\trainA\\"
        # input_data, itk_image = DICOM_read_vertical(data_path + "input_image"+str(num)+".nii")

        transducer, itk_image = makeTransducer_vertical(image_cordi, spacing[0], 71, 65,  self.trans_pos[num,:], self.target_pos)
        transducer = transducer.transpose([2, 1, 0])

        input_data = self.skull - transducer

        # #input_data = input_data.transpose([2, 1, 0])
        # trans_itk = sitk.GetImageFromArray(input_data, sitk.sitkInt8)
        # trans_itk.SetSpacing(spacing)
        # trans_itk.SetOrigin(origin)
        #
        # hlp.saveITK('input.nii', trans_itk)
        # hlp.saveITK('trnasducer.nii', itk_image)


        start = datetime.now()
        syn = self.G_A2B.predict(input_data[np.newaxis, :, :, :, np.newaxis])
        syn = np.squeeze(syn)
        end = datetime.now()
        print("prediction: ",end-start)

        start = datetime.now()
        intensity_vtk_array = ns.numpy_to_vtk(syn.flatten())
        self.image.GetPointData().SetScalars(intensity_vtk_array)
        end = datetime.now()
        print("VTK convert: ", end-start)





data_path = "F:\\syntheticGANs\\data\\T1-CT_cropped_bias_P9\\trainA\\"
input_data, itk_image = DICOM_read_vertical(data_path + "input_image1.nii")
input_data[input_data<0] = 0

spacing = itk_image.GetSpacing()
origin = itk_image.GetOrigin()  ## bounds
dimension = itk_image.GetSize()
extent = (0, dimension[0] - 1, 0, dimension[1] - 1, 0, dimension[2] - 1)

x_end = origin[0] + spacing[0] * (dimension[0] - 1)
y_end = origin[1] + spacing[1] * (dimension[1] - 1)
z_end = origin[2] + spacing[2] * (dimension[2] - 1)

x_arr = np.linspace(origin[0], x_end, dimension[0])
y_arr = np.linspace(origin[1], y_end, dimension[1])
z_arr = np.linspace(origin[2], z_end, dimension[2])

image_cordi = [x_arr, y_arr, z_arr]


input_volume, dummu = vlm.numpy_pressure_map(input_data*0, spacing, origin, dimension, extent)
skull, skull_actor = hlp.read_skull('reverse_skull-K.stl', 0.25, (0.8, 0.8, 0.8))

# A renderer and render window
ren = vtk.vtkRenderer()
ren.SetBackground(35/255, 35/255, 38/255)

points = np.array(point_ring(target, 100, 55.22))
z = points[:,1]
trans_pos = np.zeros((100,3))
trans_pos[:,0] = points[:,0]
trans_pos[:,1] = target[1]
trans_pos[:,2] = points[:,1]
# for i in range(100):
#     p, p_actor = hlp.addPoint(ren,trans_pos[i,:])
#     ren.AddActor(p_actor)


camera = vtk.vtkCamera()
camera.SetPosition(-275.7645231256627, -584.4736603963928, 99.76515581719603)
camera.SetFocalPoint(-9.580888365824716, 0.5683141846008377, -13.671290586469972)
camera.SetViewUp(0.02042422144450027, 0.18134586816821063, 0.9832072656753021)
ren.SetActiveCamera(camera)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(800, 900)

# An interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renWin)

# add the custom style
style = MouseInteractorHighLightActor(interactor, ren, input_volume, input_data)
style.SetDefaultRenderer(ren)
interactor.SetInteractorStyle(style)

axesActor = vtk.vtkAxesActor()
axesActor.SetTotalLength(5, 5, 5)
axesActor.AxisLabelsOff()

ren.AddActor(axesActor)
ren.AddActor(skull_actor)
ren.AddVolume(input_volume)

# Start
interactor.Initialize()
renWin.Render()
interactor.Start()