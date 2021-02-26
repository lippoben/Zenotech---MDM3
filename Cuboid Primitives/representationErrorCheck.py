import pandas as pd
import numpy as np


# check if a point lies within an existing cuboid. if it does then return True else return false
def inCuboid(xyz, cuboids):
    for cuboid in cuboids:
        if (cuboid[0][0] <= xyz[0] <= cuboid[1][0] and
                cuboid[0][1] <= xyz[1] <= cuboid[1][1] and
                cuboid[0][2] <= xyz[2] <= cuboid[1][2]):
            return True

    return False


# returns a boolean identifying if a point is hazardous to fly in or not
def isDangerous(xyz, dangerousArray):
    if xyz in dangerousArray:
        return True

    else:
        return False


# return true if in bounds and false if out of bounds
def checkInBounds(xyz, xyzBound):
    if xyz[0] >= xyzBound[0]:
        return False

    elif xyz[1] >= xyzBound[1]:
        return False

    elif xyz[2] >= xyzBound[2]:
        return False

    else:
        return True


nonCompressedChunkRoot = 'C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Chunks/'
compressedChunkRoot = 'C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Cuboid Primitives/compressedChunks/'


nonCompressedChunkDf = pd.read_csv(nonCompressedChunkRoot + 'Chunk1.csv')
compressedChunkDf = pd.read_csv(compressedChunkRoot + 'compressedChunk1.csv')

# define a bunch of arrays which are used in the compression process
VArray = []
xyzDangerous = []
V0Array = nonCompressedChunkDf.loc[:, 'V:0']
V1Array = nonCompressedChunkDf.loc[:, 'V:1']
V2Array = nonCompressedChunkDf.loc[:, 'V:2']
XArray = nonCompressedChunkDf.loc[:, 'Points:0']
YArray = nonCompressedChunkDf.loc[:, 'Points:1']
ZArray = nonCompressedChunkDf.loc[:, 'Points:2']
buildingFlagArray = nonCompressedChunkDf.loc[:, 'vtkValidPointMask']
# Set max safe wind speed
windSpeedDangerFlag = 5

startXYZ = np.array(compressedChunkDf.loc[:, 'startXYZ'])
endXYZ = np.array(compressedChunkDf.loc[:, 'endXYZ'])

# fetches and formats cuboid array
cuboidArray = []
for i in range(0, len(startXYZ)):
    fixedStartXYZ = startXYZ[i].strip(']')
    fixedEndXYZ = endXYZ[i].strip(']')
    fixedStartXYZ = fixedStartXYZ.strip('[')
    fixedEndXYZ = fixedEndXYZ.strip('[')
    fixedStartXYZ = fixedStartXYZ.split(',')
    fixedEndXYZ = fixedEndXYZ.split(',')
    for i in range(0, len(fixedStartXYZ)):
        fixedStartXYZ[i] = int(fixedStartXYZ[i])
        fixedEndXYZ[i] = int(fixedEndXYZ[i])

    cuboidArray.append([fixedStartXYZ, fixedEndXYZ])


# Calculate the magnitude of wind speed at every position
for i in range(0, len(V0Array)):
    VArray.append(np.sqrt(pow(V0Array[i], 2) + pow(V1Array[i], 2) + pow(V2Array[i], 2)))

# sort safe and dangerous points within the data frame
for i in range(0, len(VArray)):
    if VArray[i] > windSpeedDangerFlag or buildingFlagArray[i] == 0 and checkInBounds([int(XArray[i]), int(YArray[i]), int(ZArray[i])], [1535, 1715, 500]):
        xyzDangerous.append([int(XArray[i]), int(YArray[i]), int(ZArray[i])])


newXArray = []
newYArray = []
newZArray = []
for x in range(int(min(XArray)), int(max(XArray))+5, 5):
    newXArray.append(x)

for y in range(int(min(YArray)), int(max(YArray))+5, 5):
    newYArray.append(y)

for z in range(int(min(ZArray)), int(max(ZArray))+5, 5):
    newZArray.append(z)

XArray = newXArray
YArray = newYArray
ZArray = newZArray

missClassifiedCounter = 0

completeCounter = 1
for z in ZArray:
    print('percentage complete: ' + str(completeCounter) + '%')
    print(missClassifiedCounter)
    for y in YArray:
        for x in XArray:
            currentPoint = [int(x), int(y), int(z)]
            if isDangerous(currentPoint, xyzDangerous) and not inCuboid(currentPoint, cuboidArray):
                missClassifiedCounter += 1

    completeCounter += 1

print(missClassifiedCounter)
