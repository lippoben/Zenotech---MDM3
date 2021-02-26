import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import visualisingCuboids


# returns a boolean identifying if a point is hazardous to fly in or not
def isDangerous(xyz, dangerousArray):
    if xyz in dangerousArray:
        return True

    else:
        return False


# attempts to group cubes together into cuboids. outputs a 2d array
# format: [[bottom corner of cuboid XYZ][opposite corner of cuboid XYZ]]
def createCuboid(startXYZ, xArray, yArray, zArray):
    dimensionEndFound = False
    # iterate through all the x from the starting x until a safe cube is found. Then save last known dangerous cube as
    # the end of the x cuboid
    for X in range(startXYZ[0], XArray[-1]+5, 5):
        if not isDangerous([X, startXYZ[1], startXYZ[2]], xyzDangerous) or inCuboid([X, startXYZ[1], startXYZ[2]], cuboidArray):
            xEnd = X - 5
            dimensionEndFound = True
            break

    # if no safe cube is found then the cuboid must span the entire x dimension for this row.
    if not dimensionEndFound:
        xEnd = xArray[-1]

    # reset for Y dimension
    dimensionEndFound = False
    # iterate across each X for each Y value.
    # Only increase yEnd if all the dangerous y cubes align with shape of x cubes
    for Y in range(startXYZ[1], YArray[-1]+5, 5):
        for X in range(startXYZ[0], xEnd+5, 5):
            if not isDangerous([X, Y, startXYZ[2]], xyzDangerous) or inCuboid([X, Y, startXYZ[2]], cuboidArray):
                yEnd = Y - 5
                dimensionEndFound = True
                break

        if dimensionEndFound:
            break

    # if no safe cube is found then the cuboid must span the entire y dimension also.
    if not dimensionEndFound:
        yEnd = yArray[-1]

    # reset for Z dimension
    dimensionEndFound = False
    # iterate through X and Y limits for each Z value.
    # Only increase zEnd if all the dangerous z cubes align with shape of x and y cubes
    for Z in range(startXYZ[2], ZArray[-1]+5, 5):
        for Y in range(startXYZ[1], yEnd+5, 5):
            for X in range(startXYZ[0], xEnd+5, 5):
                if not isDangerous([X, Y, Z], xyzDangerous) or inCuboid([X, Y, Z], cuboidArray):
                    zEnd = Z - 5
                    dimensionEndFound = True
                    break
            if dimensionEndFound:
                break
        if dimensionEndFound:
            break

    # if all no safe cube is found then the entire z dimension with this x and y are dangerous so are included in cuboid
    if not dimensionEndFound:
        zEnd = zArray[-1]

    return [startXYZ, [xEnd, yEnd, zEnd]]


# check if a point lies within an existing cuboid. if it does then return True else return false
def inCuboid(xyz, cuboids):
    for cuboid in cuboids:
        if (cuboid[0][0] <= xyz[0] <= cuboid[1][0] and
                cuboid[0][1] <= xyz[1] <= cuboid[1][1] and
                cuboid[0][2] <= xyz[2] <= cuboid[1][2]):
            return True

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


chunkMax = 50

for chunkNum in range(1, chunkMax):
    print('\nStarting Compression of chunk ' + str(chunkNum))
    # read in data from preview
    df = pd.read_csv('C:/Users/lipb1/Documents/Year 3 Bristol/MDM3/Zenotech/Chunks/Chunk'+str(chunkNum)+'.csv')

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

    cuboidArray = []
    percentageCounter = 0

    for z in ZArray:
        print("Chunk compression percentage complete: ", percentageCounter, "%")
        print(len(cuboidArray))
        for y in YArray:
            for x in XArray:
                currentPoint = [x, y, z]
                if isDangerous(currentPoint, xyzDangerous) and not inCuboid(currentPoint, cuboidArray):
                    cuboidArray.append(createCuboid(currentPoint, XArray, YArray, ZArray))
        percentageCounter += 1
        if z > xyzDangerous[-1][2]:
            print("Chunk compression percentage complete: 100%")
            break

    visualisingCuboids.visualiseAndSaveCuboids('compressedChunk'+str(chunkNum), cuboidArray, [int(min(XArray)), int(max(XArray))],
                                               [int(min(YArray)), int(max(YArray))], [int(min(ZArray)), int(max(ZArray))])

'''
# visualising data
xDangerous = []
yDangerous = []
zDangerous = []
for i in range(0, len(xyzDangerous)):
    xDangerous.append(xyzDangerous[i][0])
    yDangerous.append(xyzDangerous[i][1])
    zDangerous.append(xyzDangerous[i][2])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xDangerous, yDangerous, zDangerous)
# ax.plot3D(xDangerous, yDangerous, zDangerous, 'green')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# ax.plot_wireframe(X, Y, Z, color ='green')


plt.show()
'''

'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(xSafe, ySafe, zSafe)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# ax.plot_wireframe(X, Y, Z, color ='green')


plt.show()
'''
