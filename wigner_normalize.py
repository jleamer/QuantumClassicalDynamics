from matplotlib.colors import Normalize
import numpy as np


class WignerNormalize(Normalize):
    """
    Matplotlib utility for array normalization to visualize a Wigner function

    Example of usage:
        plt.imshow(Wigner, origin='lower', norm=WignerNormalize(vmin=-0.01, vmax=0.1), cmap='seismic')
    """
    def __call__ (self, value, clip=None):
        """
        This implementation is derived from the implementation of Normalize.__call__ at
        https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)

        resdat = np.interp(result.data, (self.vmin, 0., self.vmax), (0., 0.5, 1.))
        result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result