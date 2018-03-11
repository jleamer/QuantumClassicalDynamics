from matplotlib.colors import Normalize, SymLogNorm
import numpy as np


class WignerNormalize(Normalize):
    """
    Matplotlib utility for array normalization to visualize a Wigner function.

    The main property of this normalization is that zero is map into 0.5.

    Example of usage:
        plt.imshow(Wigner, origin='lower', norm=WignerNormalize(vmin=-0.01, vmax=0.1), cmap='seismic')

    see also http://matplotlib.org/users/colormapnorms.html
    """
    def __init__(self, vmiddle=0., **kwargs):
        """
        Overloaded constructor
        :param vmiddle: The value that gets mapped into the middle color
        :param kwargs: see Normalize.__init__
        """
        super().__init__(**kwargs)
        self.vmiddle = vmiddle

    def __call__(self, value, clip=None):
        """
        This implementation is derived from the implementation of Normalize.__call__ at
        https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py
        """
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vmiddle, vmax = self.vmin, self.vmiddle, self.vmax

        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        if vmin > vmiddle or vmax < vmiddle:
            raise ValueError("middle value must be in between maxvalue and minvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)

            # in-place equivalent of above can be much faster
            resdat = np.interp(result.data, (vmin, vmiddle, vmax), (0., 0.5, 1.))
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result


class WignerSymLogNorm(SymLogNorm):
    """
    Matplotlib utility for array normalization (symmetrical logarithmic scale) to visualize a Wigner function.

    The main property of this normalization is that zero is map into 0.5.

    Example of usage:
        plt.imshow(Wigner, origin='lower', norm=WignerSymLogNorm(linthresh=1e-15, vmin=-0.01, vmax=0.1), cmap='bwr')

    see also http://matplotlib.org/users/colormapnorms.html
    """
    def __call__(self, value, clip=None):
        """
        This implementation is derived from the implementation of SymLogNorm.__call__ at
        https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py
        """
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax

        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)

            # in-place equivalent of above can be much faster
            resdat = self._transform(result.data)
            resdat = np.interp(resdat, (self._lower, 0., self._upper), (0., 0.5, 1.))
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

