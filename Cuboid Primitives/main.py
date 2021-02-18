# state file generated using paraview version 5.9.0

#### import the simple module from the paraview
from paraview.simple import *
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
renderView1.CenterOfRotation = [-250.0, 650.0, 250.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-1680.9509142204101, 813.2317829011165, 1101.3359923199264]
renderView1.CameraFocalPoint = [-250.0, 650.0, 250.0]
renderView1.CameraViewUp = [0.5062322431092422, -0.05187115528468367, 0.8608358143606814]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 433.0127018922193
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

# create a new 'Legacy VTK Reader'
canary_67p50_20p00_isosurfaces_shear_0p01vtk = LegacyVTKReader(registrationName='canary_67p50_20p00_isosurfaces_shear_0p01.vtk', FileNames=['C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives Compression/Isosurfaces/Shear/canary_67p50_20p00_isosurfaces_shear_0p01.vtk'])

# create a new 'PVD Reader'
canary_67p50_10p00_symmetrypvd = PVDReader(registrationName='canary_67p50_10p00_symmetry.pvd', FileName='C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives Compression/Isosurfaces/Volumes/canary_67p50_10p00_P10_OUTPUT/canary_67p50_10p00_symmetry.pvd')
canary_67p50_10p00_symmetrypvd.CellArrays = ['zone', 'V', 'p', 'T', 'yplus']

# create a new 'Legacy VTK Reader'
canary_67p50_10p00_isosurfaces_windspeed_1p0vtk = LegacyVTKReader(registrationName='canary_67p50_10p00_isosurfaces_windspeed_1p0.vtk', FileNames=['C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives Compression/Isosurfaces/Windspeed/canary_67p50_10p00_isosurfaces_windspeed_1p0.vtk'])

# create a new 'PVD Reader'
canary_67p50_10p00pvd = PVDReader(registrationName='canary_67p50_10p00.pvd', FileName='C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives Compression/Isosurfaces/Volumes/canary_67p50_10p00_P10_OUTPUT/canary_67p50_10p00.pvd')
canary_67p50_10p00pvd.CellArrays = ['cellvolume', 'V', 'p', 'T', 'eddy']

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=canary_67p50_10p00pvd)

# create a new 'Wavelet'
wavelet1 = Wavelet(registrationName='Wavelet1')
wavelet1.WholeExtent = [0, 50, 0, 50, 0, 50]
wavelet1.XFreq = 1.0
wavelet1.YFreq = 1.0
wavelet1.ZFreq = 1.0
wavelet1.XMag = 0.0
wavelet1.ZMag = 0.0

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=wavelet1)
transform1.Transform = 'Transform'

# init the 'Transform' selected for 'Transform'
transform1.Transform.Scale = [10.0, 10.0, 10.0]

# create a new 'Transform'
transform2 = Transform(registrationName='Transform2', Input=transform1)
transform2.Transform = 'Transform'

# init the 'Transform' selected for 'Transform'
transform2.Transform.Translate = [-500.0, 400.0, 0.0]

# create a new 'Resample With Dataset'
resampleWithDataset1 = ResampleWithDataset(registrationName='ResampleWithDataset1', SourceDataArrays=cellDatatoPointData1,
    DestinationMesh=transform2)
resampleWithDataset1.CellLocator = 'Static Cell Locator'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from resampleWithDataset1
resampleWithDataset1Display = Show(resampleWithDataset1, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
resampleWithDataset1Display.Representation = 'Wireframe'
resampleWithDataset1Display.ColorArrayName = ['POINTS', '']
resampleWithDataset1Display.SelectTCoordArray = 'None'
resampleWithDataset1Display.SelectNormalArray = 'None'
resampleWithDataset1Display.SelectTangentArray = 'None'
resampleWithDataset1Display.OSPRayScaleArray = 'cellvolume'
resampleWithDataset1Display.OSPRayScaleFunction = 'PiecewiseFunction'
resampleWithDataset1Display.SelectOrientationVectors = 'None'
resampleWithDataset1Display.ScaleFactor = 50.0
resampleWithDataset1Display.SelectScaleArray = 'cellvolume'
resampleWithDataset1Display.GlyphType = 'Arrow'
resampleWithDataset1Display.GlyphTableIndexArray = 'cellvolume'
resampleWithDataset1Display.GaussianRadius = 2.5
resampleWithDataset1Display.SetScaleArray = ['POINTS', 'cellvolume']
resampleWithDataset1Display.ScaleTransferFunction = 'PiecewiseFunction'
resampleWithDataset1Display.OpacityArray = ['POINTS', 'cellvolume']
resampleWithDataset1Display.OpacityTransferFunction = 'PiecewiseFunction'
resampleWithDataset1Display.DataAxesGrid = 'GridAxesRepresentation'
resampleWithDataset1Display.PolarAxes = 'PolarAxesRepresentation'
resampleWithDataset1Display.ScalarOpacityUnitDistance = 17.32050807568877

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
resampleWithDataset1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 125000.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
resampleWithDataset1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 125000.0, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# restore active source
SetActiveSource(resampleWithDataset1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')

