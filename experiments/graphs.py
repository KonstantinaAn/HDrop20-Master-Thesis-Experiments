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


def create_graph(x, y_CSG, y_dUSM, y_dGreedy, y_ROI, y_Winner, title):

    min_result = min(min(y_CSG), min(y_dUSM), min(y_dGreedy), min(y_ROI), min(y_Winner))
    print('Min result: ', min_result, 'and round:', round(min_result, 1))
    y_lim_left = round(min_result - 0.1, 1)

    plt.ylim(y_lim_left, 1.05)
    plt.yticks(np.arange(y_lim_left, 1.01, 0.05))

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
create_graph(x, y_CSG, y_dUSM, y_dGreedy, y_ROI, y_Winner, title)

title = 'Average ratio - Freelancer Data'
x = ['n=10, m=5','n=5, m=10','n=10, m=10','n=20, m=10','n=10, m=20','n=50, m=20','n=100, m=20','n=25, m=50','n=50, m=50','n=100, m=50','n=150, m=40']
y_CSG = [0.943304,1,0.983243,0.932576,0.95187,0.834745,0.746395,0.941865,0.83263,0.812735,0.781599]
y_dUSM = [0.956247,0.970362,0.963771,0.951255,0.91456,0.954942,0.950935,0.911031,0.906053,0.939273,0.936867]
y_dGreedy = [0.9875,1,1,0.996479,1,0.979016,0.981421,0.993909,0.983611,0.978091,0.983971]
y_ROI = [0.982143,1,0.997935,0.996479,0.999884,0.976222,0.975503,0.990216,0.977812,0.971571,0.981923]
y_Winner = [1,1,1,1,1,1,1,1,1,1,1]
create_graph(x, y_CSG, y_dUSM, y_dGreedy, y_ROI, y_Winner, title)

title = 'Average ratio - Guru Data'
x = ['n=50, m=20','n=50, m=50','n=100, m=20','n=100, m=50','n=200, m=20','n=200, m=50','n=400, m=50','n=400, m=100']
y_CSG = [0.981882,0.951035,0.958088,0.956699,0.942734,0.954908,0.914274,0.919468]
y_dUSM = [0.957051,0.938909,0.983456,0.942846,0.983167,0.9443,0.943157,0.95239]
y_dGreedy = [0.99643,0.995685,0.993334,0.991174,0.998329,0.992765,0.99408,0.994266]
y_ROI = [0.998583,0.992305,0.990137,0.988121,0.994137,0.986876,0.990644,0.994168]
y_Winner = [1,1,1,1,1,1,1,1]
create_graph(x, y_CSG, y_dUSM, y_dGreedy, y_ROI, y_Winner, title)
