from mpl_toolkits.mplot3d import Axes3D  # 这一句虽然显示灰色，但是去掉会报错。
import matplotlib.pyplot as plt

def showVoxel(voxel):
    """
    :param voxel: shape (n, n, n).
    """
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, edgecolor="k")
    plt.show()