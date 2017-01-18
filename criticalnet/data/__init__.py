from scipy.ndimage import imread
import pkg_resources


__all__ = ['lenna']

_image = {'baboon': 'baboon.png',
          'lenna': 'lenna.png'}

data_dir = pkg_resources.resource_filename('criticalnet', 'data/test/')


def sample(name='baboon'):

    return imread(data_dir.join(_image[name]), mode='F')
