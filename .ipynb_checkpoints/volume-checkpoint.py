import vtk
import numpy as np
import tkinter.filedialog as tk
import warnings
import helpfunction as hlp
import SimpleITK as sitk

warnings.filterwarnings('ignore')


from vtk.util import numpy_support as ns


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)
ren = vtk.vtkRenderer()


def nii2stl(path, save_name):

    itk_img = sitk.ReadImage(path)
    origin = itk_img.GetOrigin()  ## bounds

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(path)
    reader.Update()
    #print(reader)

    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(reader.GetOutput())
    contour.ComputeNormalsOn()
    contour.ComputeGradientsOn()
    contour.SetValue(0, 330)
    contour.Update()

    # Write in vtk
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputConnection(contour.GetOutputPort())
    triangle.PassVertsOff()
    triangle.PassLinesOff()

    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputConnection(triangle.GetOutputPort())
    decimation.Update()

    poly = decimation.GetOutput()
    poly, actor = hlp.translate(ren, origin, poly, 1, (1,1,1))

    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(save_name)
    stlWriter.SetInputData(poly)
    stlWriter.Write()


def numpy_pressure_map(pressure, spacing, origin, dimension, extent):

    vtk_pressure = ns.numpy_to_vtk(pressure.flatten())

    image = vtk.vtkImageData()
    image.SetDimensions(l2n(dimension)-1)
    image.SetExtent(extent)
    image.SetSpacing(spacing)
    image.SetOrigin(origin[0], origin[1], origin[2])
    image.GetPointData().SetScalars(vtk_pressure)

    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(image)

    opacityTransfer = vtk.vtkPiecewiseFunction()
    opacityTransfer.AddPoint(0.05,0)
    opacityTransfer.AddPoint(0.3,0.4)
    opacityTransfer.AddPoint(0.4,0.6)
    opacityTransfer.AddPoint(0.5,0.7)
    opacityTransfer.AddPoint(0.6,0.8)
    opacityTransfer.AddPoint(0.7,1)


    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(0.3, 0.1,0.1,1.0)
    ctf.AddRGBPoint(0.5, 0.2,1.0,0.2)
    ctf.AddRGBPoint(0.7, 1.0,1.0,0.0)
    ctf.AddRGBPoint(0.7, 1.0,0.0,0.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(ctf)
    volumeProperty.SetScalarOpacity(opacityTransfer)
    volumeProperty.SetScalarOpacityUnitDistance(5)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    #volume.SetOrientation(100,100,100)
    #volume.RotateX(180)
    #test = volume.GetOrientation()


    return volume, image



def pressure_map(file_name):
    grid, c2p, bounds, maximum_pressure, pointnumber,pressure_vtk  = Data_arange(file_name)



    # reader = vtk.vtkRectilinearGridReader()
    # reader.SetFileName(vtk_filename)
    # reader.Update()
    # grid = reader.GetOutput()
    bounds = grid.GetBounds()
    dimension  = grid.GetDimensions()
    extent = grid.GetExtent()
    vtk_coordi = grid.GetXCoordinates()
    xcoordi = ns.vtk_to_numpy(vtk_coordi)
    space = 1000*(xcoordi[1]- xcoordi[0])
    bounds = l2n(bounds)*1000

    image = vtk.vtkImageData()
    image.DeepCopy(grid)
    image.SetDimensions(dimension)
    image.SetExtent(extent)
    image.SetSpacing(space,space,space)
    image.SetOrigin(bounds[0], bounds[2], bounds[4])


    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(image)

    opacityTransfer = vtk.vtkPiecewiseFunction()
    opacityTransfer.AddPoint(0,0)
    opacityTransfer.AddPoint(0.15,0.9)
    opacityTransfer.AddPoint(0.3,0.99)
    opacityTransfer.AddPoint(0.5,0.99999)
    opacityTransfer.AddPoint(0.8,0.99999)
    opacityTransfer.AddPoint(0.9,0.99999999)


    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(0.3, 0.1,0.1,1.0)
    ctf.AddRGBPoint(0.5, 0.2,1.0,0.2)
    ctf.AddRGBPoint(0.8, 1.0,1.0,0.0)
    ctf.AddRGBPoint(0.9, 1.0,0.0,0.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(ctf)
    volumeProperty.SetScalarOpacity(opacityTransfer)
    volumeProperty.SetScalarOpacityUnitDistance(30)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    #volume.SetOrientation(100,100,100)
    #volume.RotateX(180)
    #test = volume.GetOrientation()


    return volume



def Data_arange(vtk_filename):

    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    grid = reader.GetOutput()
    bounds = grid.GetBounds()


    celldata = grid.GetCellData()
    pointnumber = grid.GetNumberOfPoints()
    Array0 = celldata.GetArray(0)
    Array1 = celldata.GetArray(1)

    Re = ns.vtk_to_numpy(Array0)
    Im = ns.vtk_to_numpy(Array1)

    pressure = np.absolute(Re+1j*Im)
    pressure[np.isnan(pressure)] = 0

    # pressure.reshape(138,138,192)
    # test = pressure[:,:,192/2]

    maximum_pressure = max(pressure)
    #attenuation_percentage = 0.38
    #pressure = pressure/(61351.80469*(1-attenuation_percentage)) ### maximum value SMA_K targeting normalize
    pressure = pressure/maximum_pressure ### maximum value SMA_K targeting normalize
    pressure = pressure*pressure  ##(intensity)

    count = pressure>0.9
    num_90 = np.sum(count)
    volume = num_90*0.1222*0.1222*0.1222


    pressure_vtk = ns.numpy_to_vtk(num_array=pressure,deep=True, array_type=vtk.VTK_FLOAT)

    grid.GetCellData().SetScalars(pressure_vtk)
    # writer = vtk.vtkRectilinearGridWriter()
    # writer.SetInputData(grid)
    # writer.SetFileName("result_rest900.vtk")
    # writer.Write()
    #

    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(grid)


    return grid, c2p, bounds, maximum_pressure, pointnumber, pressure_vtk



def FWHM(vtk_img, contor_percent, set_color = [1,0.3,0]):

    contour = vtk.vtkContourFilter()
    contour.SetInputData(vtk_img)
    contour.ComputeNormalsOn()
    contour.ComputeScalarsOn()
    contour.ComputeGradientsOn()
    contour.SetValue(0,contor_percent)
    contour.Update()

    transform = vtk.vtkTransform()
    transform.Scale(1,1,1)

    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputConnection(contour.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    FWHM_poly  = transform_filter.GetOutput()


    center = vtk.vtkCenterOfMass()
    center.SetInputData(FWHM_poly)
    center.Update()

    centroid_FWHM = center.GetCenter()

    mass = vtk.vtkMassProperties()
    mass.SetInputData(FWHM_poly)
    mass.Update()

    volume_FWHM = mass.GetVolume()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkLODActor()
    actor.SetNumberOfCloudPoints(200)
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(set_color)
    actor.GetProperty().SetOpacity(0.7)

    return actor, centroid_FWHM, centroid_FWHM


def iin2volume(path):

    itk_img = sitk.ReadImage(path)
    spacing = itk_img.GetSpacing()
    origin = itk_img.GetOrigin()  ## bounds
    dimension = itk_img.GetSize()
    extent =(0, dimension[0]-1, 0, dimension[1]-1, 0, dimension[2]-1)
    pressure = sitk.GetArrayFromImage(itk_img)
    m = np.max(pressure)
    pressure = pressure/15100.787
    #pressure = pressure/np.max(pressure)
    pressure_volume, vtk_img = numpy_pressure_map(pressure, spacing, origin, dimension, extent)


    return pressure_volume, vtk_img, pressure








