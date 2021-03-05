import pandas as pd
import numpy as np
import visualisingCuboids
from tqdm import tqdm
import time


# returns a boolean identifying if a point is hazardous to fly in or not
def isDangerous(xyz, dangerousArray):
    if xyz in dangerousArray:
        return True

    else:
        return False


# attempts to group cubes together into cuboids. outputs a 2d array
# format: [[bottom corner of cuboid XYZ][opposite corner of cuboid XYZ]]
def createCuboid(startXYZ, cubeStepSize):
    xEndFound = False
    yEndFound = False
    zEndFound = False

    xEnd = startXYZ[0]
    yEnd = startXYZ[1]
    zEnd = startXYZ[2]
    # while the end of the cuboid is not found search for the end of the cuboid
    while not zEndFound:
        # while the size of the plane is not found then search for the end of a the plane
        while not yEndFound:
            # while the length of rows not found then search for the end of row
            while not xEndFound:
                # if a safe cube is found or a previously represented cube is found then end row
                if ((not xEndFound and not isDangerous([xEnd + cubeStepSize, yEnd, zEnd], xyzDangerous)) or
                        inCuboid([xEnd + cubeStepSize, yEnd, zEnd], cuboidArray)):
                    xEndFound = True
                else:
                    xEnd += cubeStepSize

            # if a safe cube is found or a previously represented cube is found then end search in Y
            if not yEndFound:
                for X in range(startXYZ[0], xEnd + cubeStepSize, cubeStepSize):
                    if (not yEndFound and not isDangerous([X, yEnd + cubeStepSize, zEnd], xyzDangerous) or
                            inCuboid([X, yEnd + cubeStepSize, zEnd], cuboidArray)):
                        yEndFound = True

                if not yEndFound:
                    yEnd += cubeStepSize

        # if a safe cube is found or a previously represented cube is found then end search in Z
        if not zEndFound:
            for Y in range(startXYZ[1], yEnd + cubeStepSize, cubeStepSize):
                for X in range(startXYZ[0], xEnd + cubeStepSize, cubeStepSize):
                    if not zEndFound and not isDangerous([X, Y, zEnd + cubeStepSize], xyzDangerous):
                        zEndFound = True

            if not zEndFound:
                zEnd += cubeStepSize

    # cuboid is found so output where the cuboid starts and where the cuboid ends
    return [startXYZ, [xEnd, yEnd, zEnd]]


# check if a point lies within an existing cuboid. if it does then return True else return false
def inCuboid(xyz, cuboids):
    for cuboid in cuboids:
        if (cuboid[0][0] <= xyz[0] <= cuboid[1][0] and
                cuboid[0][1] <= xyz[1] <= cuboid[1][1] and
                cuboid[0][2] <= xyz[2] <= cuboid[1][2]):
            return True

    return False


# return true if a position lies within the bounds of sampled area return false if out of bounds else return true
def checkInBounds(xyz, xyzBound):
    if xyz[0] >= xyzBound[0]:
        return False

    elif xyz[1] >= xyzBound[1]:
        return False

    elif xyz[2] >= xyzBound[2]:
        return False

    else:
        return True


chunkMax = 50
# Set the gap between each data point
cubeSize = 5
totalVolumeMisrepresented = 0
totalStartTime = time.time()
chunkCompressionTimeArray = []
chunkCompressionRatioArray = []
for chunkNum in range(1, chunkMax):
    print('\nStarting Compression of chunk ' + str(chunkNum))
    chunkStartTime = time.time()
    # read in data from preview
    df = pd.read_csv('C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Chunks'+str(cubeSize)+F'mCubes/Chunk'+str(chunkNum)+'.csv')
    # df = pd.read_csv('C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/test.csv')

    # define a bunch of arrays which are used in the compression process
    VArray = []
    xyzDangerous = []
    xyzSafe = []
    V0Array = df.loc[:, 'V:0']
    V1Array = df.loc[:, 'V:1']
    V2Array = df.loc[:, 'V:2']
    XArray = df.loc[:, 'Points:0']
    YArray = df.loc[:, 'Points:1']
    ZArray = df.loc[:, 'Points:2']
    buildingFlagArray = df.loc[:, 'vtkValidPointMask']
    # Set max safe wind speed
    windSpeedDangerFlag = 5

    print("Flagging dangerous Cubes")
    # Calculate the magnitude of wind speed at every position and flag any position that has a dangerous wind speed or
    # is in a building
    progressBar = tqdm(total=len(V0Array))
    for i in range(0, len(V0Array)):
        magWindSpeed = np.sqrt(pow(V0Array[i], 2) + pow(V1Array[i], 2) + pow(V2Array[i], 2))
        currentCubePos = [int(XArray[i]), int(YArray[i]), int(ZArray[i])]
        if (magWindSpeed > windSpeedDangerFlag or
                (buildingFlagArray[i] == 0 and checkInBounds(currentCubePos, [1535, 1715, 500]))):
            xyzDangerous.append(currentCubePos)
        progressBar.update(1)

    progressBar.close()

    totalTrueDangerousVolume = pow(cubeSize, 3) * len(xyzDangerous)

    cuboidArray = []
    print("Dangerous points to compress: ", len(xyzDangerous))
    print("Beginning growing box algorithm")
    index = 0
    lengthDangerousArray = len(xyzDangerous)
    progressBar = tqdm(total=lengthDangerousArray)
    startTime = time.time()
    totalCuboidDangerousVolume = 0
    # Iterate through the the array of uncompressed dangerous cubes and calculate smallest number of cuboids that can
    # represent them.
    while index < lengthDangerousArray:
        xyz = [xyzDangerous[index][0], xyzDangerous[index][1], xyzDangerous[index][2]]
        if not inCuboid(xyz, cuboidArray):
            generatedCuboid = createCuboid(xyz, cubeSize)
            cuboidArray.append(generatedCuboid)
            totalCuboidDangerousVolume += (abs((generatedCuboid[1][0] + cubeSize/2) - (generatedCuboid[0][0] - cubeSize/2)) *
                     abs((generatedCuboid[1][1] + cubeSize/2) - (generatedCuboid[0][1] - cubeSize/2)) *
                     abs((generatedCuboid[1][2] + cubeSize/2) - (generatedCuboid[0][2] - cubeSize/2)))
            temp = index
            index = xyzDangerous.index([generatedCuboid[1][0], xyz[1], xyz[2]])
            progressBar.update(index-temp)

        else:
            index += 1
            progressBar.update(1)

    chunkVolumeMisrepresented = abs(totalTrueDangerousVolume - totalCuboidDangerousVolume)
    totalVolumeMisrepresented += chunkVolumeMisrepresented

    progressBar.close()
    chunkCompressionTime = time.time() - chunkStartTime
    chunkCompressionTimeArray.append(chunkCompressionTime)

    chunkCompressionRatio = len(XArray)/len(cuboidArray)
    chunkCompressionRatioArray.append(chunkCompressionRatio)
    print("Chunk compression completed!")
    print(len(xyzDangerous), "cubes compressed to ", len(cuboidArray), " cuboids")
    print("Compression time: ", chunkCompressionTime)
    print("Chunk Compression Ratio: ", chunkCompressionRatio)
    print("Total Dangerous volume to represent: ", totalTrueDangerousVolume)
    print("Total Dangerous volume represented: ", totalCuboidDangerousVolume)
    print("Total Misrepresented Dangerous Volume in Chunk: ", chunkVolumeMisrepresented)

    visualisingCuboids.visualiseAndSaveCuboids('compressedChunk' + str(chunkNum), cuboidArray,
                                               [int(min(XArray)), int(max(XArray))],
                                               [int(min(YArray)), int(max(YArray))],
                                               [int(min(ZArray)), int(max(ZArray))],
                                               cubeSize,
                                               visualise=True, save=False)

print("\nTotal time to compress all chunks: ", time.time() - totalStartTime, "s")
print("Average Chunk Compression Rate: ", np.mean(chunkCompressionTimeArray), "chunk/s")
print("Average Chunk Compression Ratio: 1:", np.mean(chunkCompressionRatioArray), " Cubes:Cuboids")
print("Total volume misrepresented: ", totalVolumeMisrepresented, "m^3")

'''
Code for visualising the uncompressed dangerous cuboids
xyzDangerousCuboidArray = []
for i in range(0, lengthDangerousArray):
    xyzDangerousCuboidArray.append([xyzDangerous[i], xyzDangerous[i]])


visualisingCuboids.visualiseAndSaveCuboids('test', xyzDangerousCuboidArray,
                                           [int(min(XArray)), int(max(XArray))],
                                           [int(min(YArray)), int(max(YArray))],
                                           [int(min(ZArray)), int(max(ZArray))],
                                           cubeSize,
                                           visualise=True, save=False)
'''