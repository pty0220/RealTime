import vtk

NUMBER_OF_SPHERES = 10


class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self, interactor):
        self.AddObserver("LeftButtonPressEvent", self.leftButtonPressEvent)
        self.AddObserver("KeyPressEvent", self.keyPressEvent)
        self.LastPickedActor = None
        self.LastPickedProperty = vtk.vtkProperty()
        self.iren = interactor

    def update(self):
        self.iren.GetRenderWindow().Render()

    def leftButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()
        print(clickPos)
        picker = vtk.vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())

        # get the new
        self.NewPickedActor = picker.GetActor()
        if self.NewPickedActor == None:
           self.NewPickedActor = self.LastPickedActor

        # If something was selected
        if self.NewPickedActor:
            # If we picked something before, reset its property
            if self.LastPickedActor:
                self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)

            # Save the property of the picked actor so that we can
            # restore it next time
            self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
            # Highlight the picked actor by changing its properties
            self.NewPickedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
            self.NewPickedActor.GetProperty().SetDiffuse(1.0)
            self.NewPickedActor.GetProperty().SetSpecular(0.0)

            # save the last picked actor
            self.LastPickedActor = self.NewPickedActor

        self.OnLeftButtonDown()
        return

        def keyPressEvent(self, obj, event):
            key = self.GetInteractor().GetKeySym()
            if key == 'Up':
                if self.NewPickedActor != None:
                    self.NewPickedActor.AddPosition(0,0,.1)
                    self.update()
                else:
                    print("not selected")

            if key == 'Left':
                if self.NewPickedActor != None:
                    self.NewPickedActor.AddPosition(0,-0.1,0)
                    self.update()
                else:
                    print("not selected")

            if key == 'Right':
                if self.NewPickedActor != None:
                    self.NewPickedActor.AddPosition(0,0.1,0)
                    self.update()
                else:
                    print("not selected")

            if key == 'Down':
                if self.NewPickedActor != None:
                    self.NewPickedActor.AddPosition(0, 0,-0.1)
                    self.update()
                else:
                    print("not selected")

            self.LastPickedActor = self.NewPickedActor
            return



# A renderer and render window
renderer = vtk.vtkRenderer()
renderer.SetBackground(.3, .4, .5)

renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(renderer)

# An interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renwin)

# add the custom style
style = MouseInteractorHighLightActor(interactor)
style.SetDefaultRenderer(renderer)
interactor.SetInteractorStyle(style)

# Add spheres to play with
for i in range(NUMBER_OF_SPHERES):
    source = vtk.vtkConeSource()

    # random position and radius
    x = vtk.vtkMath.Random(-5, 5)
    y = vtk.vtkMath.Random(-5, 5)
    z = vtk.vtkMath.Random(-5, 5)
    radius = vtk.vtkMath.Random(.5, 1.0)

    source.SetRadius(radius)
    source.SetCenter(x, y, z)
    source.SetResolution(11)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    r = vtk.vtkMath.Random(.4, 1.0)
    g = vtk.vtkMath.Random(.4, 1.0)
    b = vtk.vtkMath.Random(.4, 1.0)
    actor.GetProperty().SetDiffuseColor(r, g, b)
    actor.GetProperty().SetDiffuse(.8)
    actor.GetProperty().SetSpecular(.5)
    actor.GetProperty().SetSpecularColor(1.0, 1.0, 1.0)
    actor.GetProperty().SetSpecularPower(30.0)

    renderer.AddActor(actor)

# Start
interactor.Initialize()
interactor.Start()