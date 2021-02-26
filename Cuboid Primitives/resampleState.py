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
renderView1.CenterOfRotation = [-20.0, -11.0, 249.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-6884.584287986931, -3986.2170434546238, 4526.135456151218]
renderView1.CameraFocalPoint = [-19.999999999999982, -11.000000000000009, 249.00000000000003]
renderView1.CameraViewUp = [0.3263362895203266, 0.3716188951178577, 0.8691398178276545]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 2332.514737359659
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layoutss
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

# create a new 'Wavelet'
wavelet1 = Wavelet(registrationName='Wavelet1')
wavelet1.WholeExtent = [0, 50, 0, 50, 0, 50]

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=wavelet1)
transform1.Transform = 'Transform'

# init the 'Transform' selected for 'Transform'
transform1.Transform.Scale = [10.0, 10.0, 10.0]

# create a new 'PVD Reader'
canary_67p50_10p00pvd = PVDReader(registrationName='canary_67p50_10p00.pvd', FileName='C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives/Isosurfaces/Volumes/canary_67p50_10p00.pvd')
canary_67p50_10p00pvd.CellArrays = ['cellvolume', 'V', 'p', 'T', 'eddy']

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=canary_67p50_10p00pvd)
cellDatatoPointData1.CellDataArraytoprocess = ['T', 'V', 'cellvolume', 'eddy', 'p']

# create a new 'Resample With Dataset'
resampleWithDataset1 = ResampleWithDataset(registrationName='ResampleWithDataset1', SourceDataArrays=cellDatatoPointData1,
    DestinationMesh=transform1)
resampleWithDataset1.CellLocator = 'Static Cell Locator'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from cellDatatoPointData1
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'cellvolume'
cellvolumeLUT = GetColorTransferFunction('cellvolume')
cellvolumeLUT.RGBPoints = [30.517578125, 0.231373, 0.298039, 0.752941, 62515.2587890625, 0.865003, 0.865003, 0.865003, 125000.0, 0.705882, 0.0156863, 0.14902]
cellvolumeLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'cellvolume'
cellvolumePWF = GetOpacityTransferFunction('cellvolume')
cellvolumePWF.Points = [30.517578125, 0.0, 0.5, 0.0, 125000.0, 1.0, 0.5, 0.0]
cellvolumePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
cellDatatoPointData1Display.Representation = 'Surface'
cellDatatoPointData1Display.ColorArrayName = ['POINTS', 'cellvolume']
cellDatatoPointData1Display.LookupTable = cellvolumeLUT
cellDatatoPointData1Display.Opacity = 0.4
cellDatatoPointData1Display.SelectTCoordArray = 'None'
cellDatatoPointData1Display.SelectNormalArray = 'None'
cellDatatoPointData1Display.SelectTangentArray = 'None'
cellDatatoPointData1Display.OSPRayScaleArray = 'cellvolume'
cellDatatoPointData1Display.OSPRayScaleFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.SelectOrientationVectors = 'None'
cellDatatoPointData1Display.ScaleFactor = 345.0
cellDatatoPointData1Display.SelectScaleArray = 'cellvolume'
cellDatatoPointData1Display.GlyphType = 'Arrow'
cellDatatoPointData1Display.GlyphTableIndexArray = 'cellvolume'
cellDatatoPointData1Display.GaussianRadius = 17.25
cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'cellvolume']
cellDatatoPointData1Display.ScaleTransferFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.OpacityArray = ['POINTS', 'cellvolume']
cellDatatoPointData1Display.OpacityTransferFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.DataAxesGrid = 'GridAxesRepresentation'
cellDatatoPointData1Display.PolarAxes = 'PolarAxesRepresentation'
cellDatatoPointData1Display.ScalarOpacityFunction = cellvolumePWF
cellDatatoPointData1Display.ScalarOpacityUnitDistance = 26.649014913335645
cellDatatoPointData1Display.OpacityArrayName = ['POINTS', 'cellvolume']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cellDatatoPointData1Display.ScaleTransferFunction.Points = [30.517578125, 0.0, 0.5, 0.0, 125000.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cellDatatoPointData1Display.OpacityTransferFunction.Points = [30.517578125, 0.0, 0.5, 0.0, 125000.0, 1.0, 0.5, 0.0]

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

# setup the color legend parameters for each legend in this view

# get color legend/bar for cellvolumeLUT in view renderView1
cellvolumeLUTColorBar = GetScalarBar(cellvolumeLUT, renderView1)
cellvolumeLUTColorBar.Title = 'cellvolume'
cellvolumeLUTColorBar.ComponentTitle = ''

# set color bar visibility
cellvolumeLUTColorBar.Visibility = 1

# show color legend
cellDatatoPointData1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(resampleWithDataset1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')