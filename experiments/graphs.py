from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#---------------3d graph-----------------
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
#
# xpos = [1,2,3,4,5,6,7,8,9,10]
# ypos = [2,3,4,5,1,6,2,1,7,2]
# num_elements = len(xpos)
# zpos = [0,0,0,0,0,0,0,0,0,0]
# dx = np.ones(10)
# dy = np.ones(10)
# dz = [1,2,3,4,5,6,7,8,9,10]
#
# ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
# plt.show()

# create data

#---------------------2d graph---------------------

def createGraph(x, y_CSG, y_dUSM, y_dGreedy, y_ROI, y_Winner, title):

    plt.ylim(0.75, 1.05)
    plt.yticks(np.arange(0.75, 1.01, 0.05))

    plt.plot(x, y_CSG, 'g', linewidth=2, label="CSG")
    plt.plot(x, y_dUSM, 'k', linewidth=2, label="dUSM")
    plt.plot(x, y_dGreedy, color='orange', linewidth=2, label="dGreedy")
    plt.plot(x, y_ROI, 'b--', linewidth=2, label="ROI")
    plt.plot(x, y_Winner, 'm', linewidth=2, label="Winner")
    plt.title(title)

    plt.legend()
    plt.show()


title = 'Average ratio - Synthetic Data'
x = ['n=10, m=5','n=5, m=10','n=10, m=10','n=20, m=10','n=10, m=20','n=50, m=20','n=100, m=20','n=25, m=50','n=50, m=50','n=100, m=50']
y_CSG = [0.842295, 0.95962, 0.9294466, 0.93051, 0.92833, 0.96377, 0.9755, 0.97286, 0.95743, 0.957428]
y_dUSM = [0.96441,0.8635,0.9188,0.95804,0.85847,0.93464,0.93068,0.82416,0.85124,0.87694]
y_dGreedy = [0.99224, 0.99801, 0.98586,	0.99309, 0.98004, 0.98973, 0.99453,	0.99011, 0.98043, 0.98573]
y_ROI = [0.99313, 0.98628, 0.98686,	0.99217, 0.97817, 0.99076, 0.99422, 0.97946, 0.99029, 0.99005]
y_Winner = [1,	1,	1,	1,	0.99961, 0.99949, 0.99907, 0.99902, 0.99317, 0.99605]
createGraph(x, y_CSG, y_dUSM, y_dGreedy, y_ROI, y_Winner, title)

