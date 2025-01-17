from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as colours
import csv


def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data2(p, size=s))
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors, 6), **kwargs)


def visualiseAndSaveCuboids(chunkName, cuboidArray, xBounds, yBounds, zBounds, cubeSize, save=True, visualise=True):

    if visualise:
        positions = []
        sizes = []
        for cuboid in cuboidArray:
            positions.append([cuboid[0][0] - (cubeSize/2), cuboid[0][1] - (cubeSize/2), cuboid[0][2] - (cubeSize/2)])
            lengthX = (cuboid[1][0] - cuboid[0][0]) + cubeSize
            lengthY = (cuboid[1][1] - cuboid[0][1]) + cubeSize
            lengthZ = (cuboid[1][2] - cuboid[0][2]) + cubeSize
            sizes.append([lengthX, lengthY, lengthZ])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        pc = plotCubeAt2(positions, sizes, edgecolor="k", colors=colours)
        ax.add_collection3d(pc)

        xBounds[0] -= cubeSize
        zBounds[0] -= cubeSize
        yBounds[0] -= cubeSize
        xBounds[1] += cubeSize
        zBounds[1] += cubeSize
        yBounds[1] += cubeSize

        ax.set_xlim(xBounds)
        ax.set_ylim(yBounds)
        ax.set_zlim(zBounds)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    if save:
        header = ['startXYZ', 'endXYZ']
        with open('compressedChunks'+str(cubeSize)+'mCubes/'+chunkName+'.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(header)

            for row in cuboidArray:
                writer.writerow(row)

            f.close()


