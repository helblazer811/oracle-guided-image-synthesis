import io
import numpy as np
import matplotlib.pyplot as plt


def plot_to_numpy(fig):
    #DPI = 500
    io_buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(io_buf, format='raw')#, dpi=DPI)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                 newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


