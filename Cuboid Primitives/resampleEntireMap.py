# state file generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
import time
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1508, 796]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [225.00000000000034, 237.00000000000045, 249.50000000000057]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-103.37081088364063, 4890.766585233412, 2303.7775604604344]
renderView1.CameraFocalPoint = [-1212.6071818952748, 377.08534090274327, 972.2238063782921]
renderView1.CameraViewUp = [-3.095093775387636e-05, -0.28294162692192554, 0.9591371303399808]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 2682.43904870176
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1508, 796)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

cubeSize = 4.0
# create a new 'Wavelet'
wavelet1 = Wavelet(registrationName='Wavelet1')
wavelet1.WholeExtent = [0, 125, 0, 125, 0, 125]
wavelet1.XFreq = 10.0
wavelet1.YFreq = 10.0
wavelet1.ZFreq = 10.0
wavelet1.XMag = 0.0
wavelet1.YMag = 0.0
wavelet1.ZMag = 0.0

# create a new 'PVD Reader'
canary_135p00_15p00pvd = PVDReader(registrationName='canary_135p00_15p00.pvd', FileName='C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives/Isosurfaces/Volumes/canary_135p00_15p00_P10_OUTPUT/canary_135p00_15p00.pvd')
canary_135p00_15p00pvd.CellArrays = ['cellvolume', 'V', 'p', 'T', 'eddy']

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=canary_135p00_15p00pvd)
cellDatatoPointData1.CellDataArraytoprocess = ['T', 'V', 'cellvolume', 'eddy', 'p']

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=wavelet1)
transform1.Transform = 'Transform'

# init the 'Transform' selected for 'Transform'
transform1.Transform.Scale = [cubeSize, cubeSize, cubeSize]

# create a new 'Transform'
transform2 = Transform(registrationName='Transform2', Input=transform1)
transform2.Transform = 'Transform'


root = 'C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Chunks4mCubes/'
chunkCounter = 1
for Y in range(-1735, 1765, int(wavelet1.WholeExtent[1] * transform1.Transform.Scale[0])):
    for X in range(-1570, 1930, int(wavelet1.WholeExtent[3] * transform1.Transform.Scale[1])):
        start_time = time.time()
        transform2.Transform.Translate = [X, Y, 0.0]
        transform2.Transform.Scale = [1.0, 1.0, 1.0]
        regName = 'ResampleWithDataset'
        # create a new 'Resample With Dataset'
        resampleWithDataset1 = ResampleWithDataset(registrationName=regName,
                                                   SourceDataArrays=cellDatatoPointData1,
                                                   DestinationMesh=transform2)
        resampleWithDataset1.CellLocator = 'Static Cell Locator'

        sampleName = root + 'Chunk' + str(chunkCounter) + '.csv'
        SaveData(sampleName, resampleWithDataset1, Precision=5, FieldAssociation='Point Data')
        chunkCounter += 1
        print("Chunk sample time: ", time.time() - start_time)


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
