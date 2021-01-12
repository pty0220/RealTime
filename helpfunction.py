import vtk
import numpy as np
from vtk.util import numpy_support as ns
import SimpleITK as sitk


l2n = lambda l: np.array(l)
n2l = lambda n: list(n)

def makeSphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    result = (arr <= 1.0)
    result = result.astype(int)

    return result

def makeTransducer_vertical(image_cordi, step_size, ROC, width, tran_pose, target_pose):

    ## ROC depend on Voxel
    radiusVN = int(np.round(ROC / step_size))
    widthVN = int(np.round(width / step_size))

    ## direction vector from target to geometrical focus
    dirVec = (target_pose-tran_pose)/np.linalg.norm(target_pose-tran_pose)
    geometry_center = target_pose+dirVec*np.abs(ROC-np.linalg.norm(target_pose-tran_pose))

    x_arr = image_cordi[0]
    y_arr = image_cordi[1]
    z_arr = image_cordi[2]

    x_idx = np.argmin(np.abs(x_arr-geometry_center[0]))
    y_idx = np.argmin(np.abs(y_arr-geometry_center[1]))
    z_idx = np.argmin(np.abs(z_arr-geometry_center[2]))

    # Make mesh grid, order is VERY IMPORTANT!!
    my, mx, mz = np.meshgrid(y_arr, x_arr, z_arr)
    shape = mx.shape

    # Make large sphere for transducer
    sphere = makeSphere((shape[0], shape[1], shape[2]), radiusVN, (x_idx, y_idx, z_idx))
    sphere_sub = makeSphere((shape[0], shape[1], shape[2]), radiusVN-1, (x_idx, y_idx, z_idx))
    sphere = sphere - sphere_sub

    # Make plane to cut the sphere and make transducer
    # calculate normal vector of plane
    normal = (tran_pose-target_pose)/np.linalg.norm(tran_pose-target_pose)
    temp_length = np.sqrt(ROC**2 - (width/2)**2)

    # Find one point on the cutting plane
    plane_point = geometry_center + temp_length*normal

    # Plane equation using normal and point
    cont = normal[0]*plane_point[0] + normal[1]*plane_point[1] + normal[2]*plane_point[2]
    plane_cal = mx * normal[0] + my * normal[1] + mz * normal[2] - cont
    plane_cal = np.round(plane_cal, 1)

    plane_cal[plane_cal > 0] = 1
    plane_cal[plane_cal < 0] = 0
    plane_cal = plane_cal.astype(int)

    # Cut sphere using plane
    transducer = np.multiply(sphere, plane_cal)

    return transducer


def live_make_volume(vtkImage, value):
    intensity_vtk_pointdata = vtkImage.GetPointData()
    intensity_vtk_array = intensity_vtk_pointdata.GetArray(0)
    intensity_np_array = ns.vtk_to_numpy(intensity_vtk_array)
    intensity_np_array =intensity_np_array + intensity_np_array*value
    intensity_vtk_array = ns.numpy_to_vtk(intensity_np_array)
    vtkImage.GetPointData().SetScalars(intensity_vtk_array)


def make_volume(vtkImage):
    intensity_vtk_pointdata = vtkImage.GetPointData()
    intensity_vtk_array = intensity_vtk_pointdata.GetArray(0)
    intensity_np_array = ns.vtk_to_numpy(intensity_vtk_array)
    maxI = intensity_np_array.max()
    max = intensity_np_array.max()
    intensity_np_array =intensity_np_array/maxI
    intensity_vtk_array = ns.numpy_to_vtk(intensity_np_array)
    vtkImage.GetPointData().SetScalars(intensity_vtk_array)

    vtkImage.GetDemensions()

    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(vtkImage)

    ## for rendering
    opacityTransfer = vtk.vtkPiecewiseFunction()
    opacityTransfer.AddPoint(0, 0)
    opacityTransfer.AddPoint(0.05, 0.9999)
    opacityTransfer.AddPoint(0.15, 0.999999)
    # opacityTransfer.AddPoint(0.8,0.6)
    opacityTransfer.AddPoint(0.3, 0.99999999999)

    ctf = vtk.vtkColorTransferFunction()
    ctf.AddRGBPoint(0.05, 0.1, 0.1, 1.0)
    ctf.AddRGBPoint(0.15, 0.2, 1.0, 0.2)
    ctf.AddRGBPoint(0.5, 1.0, 0.0, 0.0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(ctf)
    volumeProperty.SetScalarOpacity(opacityTransfer)
    volumeProperty.SetScalarOpacityUnitDistance(300)

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume


def read_skull_vtk(filename,  opacity):
    ren = vtk.vtkRenderer()




    readerstl = vtk.vtkDataSetReader()
    readerstl.SetFileName(filename)
    readerstl.Update()

    reader = readerstl.GetOutput()


    STLmapper = vtk.vtkPolyDataMapper()
    STLmapper.SetInputData(reader)

    STLactor = vtk.vtkActor()
    STLactor.SetMapper(STLmapper)
    STLactor.GetProperty().SetColor(0.4,0.4,0.4)
    STLactor.GetProperty().SetOpacity(opacity)

    return reader, STLactor




def read_skull(filename,  opacity, color):
    ren = vtk.vtkRenderer()




    readerstl = vtk.vtkSTLReader()
    readerstl.SetFileName(filename)
    readerstl.Update()

    reader = readerstl.GetOutput()


    STLmapper = vtk.vtkPolyDataMapper()
    STLmapper.SetInputData(reader)

    STLactor = vtk.vtkActor()
    STLactor.SetMapper(STLmapper)
    STLactor.GetProperty().SetOpacity(opacity)
    STLactor.GetProperty().SetColor(color)




    return reader, STLactor


def addLine(ren, p1, p2, color=[0.0, 0.0, 1.0], opacity=1.0):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty()

    return line, actor


def addPoint(ren, p, color=[0.0,0.0,0.0], radius=0.5):

    point = vtk.vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(1)



    return point, actor


def make_centerline_target(skull, target, centerline_length):
    pointLocator = vtk.vtkPointLocator()
    pointLocator.SetDataSet(skull)
    pointLocator.BuildLocator()
    id = pointLocator.FindClosestPoint(target)
    point_s = skull.GetPoint(id)

    vector = l2n(point_s) - l2n(target)
    centerline_vector = vector / np.linalg.norm(vector)
    centerline_target = n2l(l2n(target) + centerline_length * centerline_vector )
    middle_target =  n2l(l2n(target) - 10 * centerline_vector )
    deep_target = n2l(l2n(target) - 20 * centerline_vector )
    return centerline_target, centerline_vector , point_s, middle_target, deep_target


def make_analysis_range2(num_pts, radius, range_angle, opacity=0.25,centerline_vector=[0,0,0], Target=[0,0,0]):

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
    coordi[:, 0] = x
    coordi[:, 1] = y
    coordi[:, 2] = z


    points = vtk.vtkPoints()

    for i in range(len(x)):
        points.InsertNextPoint(x[i],y[i],z[i])
        dis = np.sqrt(np.sum(np.power((coordi-coordi[i,:]),2), axis =1))
        dis_min[i] = np.min(dis[dis>0])


    dis_average = np.average(dis_min)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData( poly ) # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection( d3D.GetOutputPort() )
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()


    # rotation of analysis range
    center_vector = [1, 0, 0]
    unit_vector = centerline_vector / np.linalg.norm(centerline_vector)
    xy_unit_vector = l2n((unit_vector[0], unit_vector[1], 0))

    if n2l(xy_unit_vector) == [0, 0, 0]:
        xy_angle = 0.0
        z_angle = 90.0
    else:
        xy_angle = np.rad2deg(np.arccos(
            np.dot(center_vector, xy_unit_vector) / (np.linalg.norm(center_vector) * np.linalg.norm(xy_unit_vector))))
        z_angle = np.rad2deg(np.arccos(
            np.dot(xy_unit_vector, unit_vector) / (np.linalg.norm(xy_unit_vector) * np.linalg.norm(unit_vector))))
    if unit_vector[2] < 0:
        z_angle = -z_angle
    if unit_vector[1] < 0:
        xy_angle = -xy_angle

    #### transform (rotation)
    ##### translate first !!!! rotate second !!!!!!!!!!!!!!! important!!!!!

    transform = vtk.vtkTransform()
    transform.Translate(Target)

    transform.RotateWXYZ(90, 0, 1, 0)
    transform.RotateWXYZ(-xy_angle, 1, 0, 0)
    transform.RotateWXYZ(-z_angle, 0, 1, 0)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(spherePoly)
    transformFilter.Update()

    Cutpoly = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    # for debugging dummy file
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(spherePoly)



    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor([0,0,1])
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetEdgeColor([0,0,0])
    actor.SetMapper(mapper)

    return Cutpoly, actor, dis_average, dis_min


def make_evencirle(num_pts=1000, ROC=71, width=65, focal_length=55.22, range_vector=[0, 0, 0], Target=[0, 0, 0],
                   opacity=0.7, color=[1, 0, 0]):
    ##################### make transducer function with evely distributed spots

    X = Target[0]
    Y = Target[1]
    Z = Target[2]

    h_wid = width / 2
    p_height = ROC ** 2 - h_wid ** 2
    height_from_center = np.sqrt(p_height)
    height = ROC - height_from_center  ### transducer's height
    rate = height / (ROC * 2)  ## ratio height/ROC*2

    indices_theta = np.arange(0, num_pts, dtype=float)
    indices_phi = np.linspace(0, num_pts * rate, num=num_pts)  ## define transdcuer's height as ratio

    phi = np.arccos(1 - 2 * indices_phi / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices_theta

    phi_deg = np.rad2deg(phi)




    x, y, z = np.cos(theta) * np.sin(phi) * ROC + X, np.sin(theta) * np.sin(phi) * ROC + Y, np.cos(
        phi) * ROC + Z;


    # for mesh distance calculation
    coordi  = np.zeros((num_pts, 3))
    dis_min = np.zeros((num_pts,1))
    coordi[:, 0] = x
    coordi[:, 1] = y
    coordi[:, 2] = z



    # x,y,z is coordination of evenly distributed shpere
    # I will try to make poly data use this x,y,z*radius


    points = vtk.vtkPoints()

    for i in range(len(x)):
        points.InsertNextPoint(x[i], y[i], z[i])
        dis = np.sqrt(np.sum(np.power((coordi-coordi[i,:]),2), axis =1))
        dis_min[i] = np.min(dis[dis>0])

    dis_average = np.average(dis_min)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData(poly)  # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection(d3D.GetOutputPort())
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()

    return spherePoly, dis_average, dis_min

def make_transducer(spherePoly, ROC=71, width=65, focal_length=55.22, range_vector=[0, 0, 0], Target=[0, 0, 0],
                    opacity=1, color=[1, 0, 0]):


    center_vector = [1, 0, 0]
    unit_vector = range_vector / np.linalg.norm(range_vector)
    xy_unit_vector = l2n((unit_vector[0], unit_vector[1], 0))

    if n2l(xy_unit_vector) == [0, 0, 0]:
        xy_angle = 0.0
        z_angle = 90.0
    else:
        xy_angle = np.rad2deg(np.arccos(
            np.dot(center_vector, xy_unit_vector) / (np.linalg.norm(center_vector) * np.linalg.norm(xy_unit_vector))))
        z_angle = np.rad2deg(np.arccos(
            np.dot(xy_unit_vector, unit_vector) / (np.linalg.norm(xy_unit_vector) * np.linalg.norm(unit_vector))))
    if unit_vector[2] < 0:
        z_angle = -z_angle
    if unit_vector[1] < 0:
        xy_angle = -xy_angle

    #### transform (rotation)


    gap = focal_length - ROC
    GAP = n2l(l2n(Target) + l2n(range_vector) * gap)



    transform = vtk.vtkTransform()
    transform.Translate(GAP)    #### move to the gap(trandcuer center to target) point
    transform.RotateWXYZ(90, 0, 1, 0)
    transform.RotateWXYZ(-xy_angle, 1, 0, 0)
    transform.RotateWXYZ(-z_angle, 0, 1, 0)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(spherePoly)
    transformFilter.Update()

    Transducer = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)


    return Transducer, actor, xy_angle, z_angle

def saveITK(path, itkImage):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(itkImage)

def cubeaxes(bounds):
    ren = vtk.vtkRenderer()
    ####### make grid
    cubeAxesActor = vtk.vtkCubeAxesActor()
    cubeAxesActor.SetBounds(bounds)
    cubeAxesActor.SetCamera(ren.GetActiveCamera())
    cubeAxesActor.GetTitleTextProperty(0).SetColor(1.0,0.0,0.0)
    cubeAxesActor.GetLabelTextProperty(0).SetColor(1.0,0.0,0.0)

    cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0,1.0,0.0)
    cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0,1.0,0.0)

    cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0,0.0,1.0)
    cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0,0.0,1.0)

    cubeAxesActor.XAxisLabelVisibilityOff()
    cubeAxesActor.YAxisLabelVisibilityOff()
    cubeAxesActor.ZAxisLabelVisibilityOff()


    cubeAxesActor.DrawXGridlinesOn()
    cubeAxesActor.DrawYGridlinesOn()
    cubeAxesActor.DrawZGridlinesOn()

    #cubeAxesActor.SetGridLineLocation(vtk.VTK_GRID_LINES_FURTHEST)

    cubeAxesActor.SetGridLineLocation(cubeAxesActor.VTK_GRID_LINES_FURTHEST)
    cubeAxesActor.XAxisMinorTickVisibilityOff()
    cubeAxesActor.YAxisMinorTickVisibilityOff()
    cubeAxesActor.ZAxisMinorTickVisibilityOff()

    cubeAxesActor.GetXAxesLinesProperty().SetColor(1,1,1)
    cubeAxesActor.GetYAxesLinesProperty().SetColor(1, 1, 1)
    cubeAxesActor.GetZAxesLinesProperty().SetColor(1, 1, 1)

    cubeAxesActor.GetXAxesGridlinesProperty().SetColor(1,1,1)
    cubeAxesActor.GetYAxesGridlinesProperty().SetColor(1,1,1)
    cubeAxesActor.GetZAxesGridlinesProperty().SetColor(1,1,1)
    cubeAxesActor.RotateX(90)
    return cubeAxesActor


def cut_skull(skull,Target, centerline_vector,opacity=0.3):

    cutting_center = n2l(l2n(Target) - l2n(centerline_vector)*15)

    Plane = vtk.vtkPlane()
    Plane.SetOrigin(cutting_center)
    Plane.SetNormal(centerline_vector)



    source = vtk.vtkCylinder()
    source.SetCenter(Target)
    source.SetRadius(50.0)

    Clipper = vtk.vtkClipPolyData()
    Clipper.SetInputData(skull)
    Clipper.SetClipFunction(Plane)
    Clipper.SetValue(0);
    Clipper.Update()

    skull_cut = Clipper.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(skull_cut)

    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetColor([0.45, 0.45, 0.45])
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetEdgeColor([0, 0, 0])
    actor.SetMapper(mapper)

    return skull_cut, actor


def cut_skull_loop(skull,Target, centerline_vector,opacity=0.7):
    cutting_center = n2l(l2n(Target) + l2n(centerline_vector)*5)

    Plane = vtk.vtkPlane()
    Plane.SetOrigin(Target)
    Plane.SetNormal(centerline_vector)



    source = vtk.vtkCylinder()
    source.SetCenter(Target)
    source.SetRadius(50.0)

    Clipper = vtk.vtkClipPolyData()
    Clipper.SetInputData(skull)
    Clipper.SetClipFunction(Plane)
    Clipper.SetValue(0);
    Clipper.Update()

    skull_cut = Clipper.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(skull_cut)

    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor([0.4, 0.4, 0.4])
    actor.GetProperty().EdgeVisibilityOff()
    actor.GetProperty().SetEdgeColor([0, 0, 0])
    actor.SetMapper(mapper)

    return skull_cut, actor


def translate (ren,move_point ,vtkobject,opacity, color ):

    transform = vtk.vtkTransform()
    #transform.RotateWXYZ(90, 0, 1, 0)
    transform.Translate(move_point)    #### move to the gap(trandcuer center to target) point

    #transform.RotateWXYZ(67.24592159420198811, 0, 1, 0)
    #transform.RotateWXYZ(-6.23172765980718246, 1, 0, 0)

    #transform.Translate(move_point)    #### move to the gap(trandcuer center to target) point


    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)

    if vtkobject.GetClassName() == 'vtkPolyData':
        transformFilter.SetInputData(vtkobject)
    else:
        transformFilter.SetInputConnection(vtkobject.GetOutputPort())

    transformFilter.Update()

    move_poly = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().EdgeVisibilityOff()

    return move_poly, actor


def rotate (ren,vector,angle ,vtkobject,opacity, color ):

    transform = vtk.vtkTransform()
    #transform.RotateWXYZ(90, 0, 1, 0)
   #transform.Translate(move_point)    #### move to the gap(trandcuer center to target) point
    transform.RotateWXYZ(angle, vector[0], vector[1], vector[2])


    #transform.RotateWXYZ(67.24592159420198811, 0, 1, 0)
    #transform.RotateWXYZ(90, 0, 0, 1)
    #transform.Translate(move_point)    #### move to the gap(trandcuer center to target) point


    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)

    if vtkobject.GetClassName() == 'vtkPolyData':
        transformFilter.SetInputData(vtkobject)
    else:
        transformFilter.SetInputConnection(vtkobject.GetOutputPort())

    transformFilter.Update()

    move_poly = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().EdgeVisibilityOff()

    return move_poly, actor




##### this is old version cut sphere method
def make_analysis_rage(num_pts, radius, opacity=0.25, centerline_vector=[0, 0, 0], Target=[0, 0, 0]):
    X = Target[0]
    Y = Target[1]
    Z = Target[2]

    indices = np.arange(0, num_pts, dtype=float) + 0.5
    radius
    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi) * radius + X, np.sin(theta) * np.sin(phi) * radius + Y, np.cos(
        phi) * radius + Z;
    coordi = np.zeros((num_pts, 3))
    # x,y,z is coordination of evenly distributed shpere
    # I will try to make poly data use this x,y,z*radius

    points = vtk.vtkPoints()

    for i in range(len(x)):
        array_point = np.array([x[i], y[i], z[i]])
        points.InsertNextPoint(x[i], y[i], z[i])
        coordi[i][0] = x[i]
        coordi[i][1] = y[i]
        coordi[i][2] = z[i]

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData(poly)  # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection(d3D.GetOutputPort())
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()

    # calculate set point of analysis area
    cutting_center = n2l(l2n(Target) + l2n(centerline_vector) * (l2n(radius) * (np.sqrt(2) / 2)))

    Plane = vtk.vtkPlane()
    Plane.SetOrigin(cutting_center)
    Plane.SetNormal(centerline_vector)

    Clipper = vtk.vtkClipPolyData()
    Clipper.SetInputData(spherePoly)
    Clipper.SetClipFunction(Plane)
    Clipper.SetValue(0);
    Clipper.Update()

    Cutpoly = Clipper.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(Cutpoly)

    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor([0, 0, 1])
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor([0, 0, 0])
    actor.SetMapper(mapper)

    return Cutpoly, actor, coordi


    ##### this is old version cut sphere method


def make_analysis_rage_test(num_pts, radius, range_angle, opacity=0.25,centerline_vector=[0,0,0], Target=[0,0,0]):

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
    coordi = np.zeros((num_pts, 3))


    points = vtk.vtkPoints()

    for i in range(len(x)):
        points.InsertNextPoint(x[i],y[i],z[i])
        coordi[i][0] = x[i]
        coordi[i][1] = y[i]
        coordi[i][2] = z[i]
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData( poly ) # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection( d3D.GetOutputPort() )
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()


    # rotation of analysis range
    center_vector = [1, 0, 0]
    unit_vector = centerline_vector / np.linalg.norm(centerline_vector)
    xy_unit_vector = l2n((unit_vector[0], unit_vector[1], 0))

    if n2l(xy_unit_vector) == [0, 0, 0]:
        xy_angle = 0.0
        z_angle = 90.0
    else:
        xy_angle = np.rad2deg(np.arccos(
            np.dot(center_vector, xy_unit_vector) / (np.linalg.norm(center_vector) * np.linalg.norm(xy_unit_vector))))
        z_angle = np.rad2deg(np.arccos(
            np.dot(xy_unit_vector, unit_vector) / (np.linalg.norm(xy_unit_vector) * np.linalg.norm(unit_vector))))
    if unit_vector[2] < 0:
        z_angle = -z_angle
    if unit_vector[1] < 0:
        xy_angle = -xy_angle

    #### transform (rotation)
    ##### translate first !!!! rotate second !!!!!!!!!!!!!!! important!!!!!

    transform = vtk.vtkTransform()
    transform.Translate(Target)

    transform.RotateWXYZ(90, 0, 1, 0)
    transform.RotateWXYZ(-xy_angle, 1, 0, 0)
    transform.RotateWXYZ(-z_angle, 0, 1, 0)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(spherePoly)
    transformFilter.Update()

    Cutpoly = transformFilter.GetOutput()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(transformFilter.GetOutput())

    # for debugging dummy file
    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetInputData(spherePoly)



    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor([0,0,1])
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor([0,0,0])
    actor.SetMapper(mapper)

    return Cutpoly, actor, coordi


def make_evencirle_test(num_pts=1000, ROC=71, width=65, focal_length=55.22, range_vector=[0, 0, 0], Target=[0, 0, 0],
                   opacity=0.7, color=[1, 0, 0]):
    ##################### make transducer function with evely distributed spots

    X = Target[0]
    Y = Target[1]
    Z = Target[2]

    h_wid = width / 2
    p_height = ROC ** 2 - h_wid ** 2
    height_from_center = np.sqrt(p_height)
    height = ROC - height_from_center  ### transducer's height
    rate = height / (ROC * 2)  ## ratio height/ROC*2

    indices_theta = np.arange(0, num_pts, dtype=float)
    indices_phi = np.linspace(0, num_pts * rate, num=num_pts)  ## define transdcuer's height as ratio

    phi = np.arccos(1 - 2 * indices_phi / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices_theta

    phi_deg = np.rad2deg(phi)

    x, y, z = np.cos(theta) * np.sin(phi) * ROC + X, np.sin(theta) * np.sin(phi) * ROC + Y, np.cos(
        phi) * ROC + Z;

    # x,y,z is coordination of evenly distributed shpere
    # I will try to make poly data use this x,y,z*radius

    points = vtk.vtkPoints()

    for i in range(len(x)):
        points.InsertNextPoint(x[i], y[i], z[i])

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    # To create surface of a sphere we need to use Delaunay triangulation
    d3D = vtk.vtkDelaunay3D()
    d3D.SetInputData(poly)  # This generates a 3D mesh

    # We need to extract the surface from the 3D mesh
    dss = vtk.vtkDataSetSurfaceFilter()
    dss.SetInputConnection(d3D.GetOutputPort())
    dss.Update()

    # Now we have our final polydata
    spherePoly = dss.GetOutput()

    return spherePoly