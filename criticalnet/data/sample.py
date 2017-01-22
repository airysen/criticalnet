from scipy.ndimage import imread
import pkg_resources


_image = {'baboon': 'baboon.png',
          'lenna': 'lenna.png'}

data_dir = pkg_resources.resource_filename('criticalnet', '/data/test/')


def sample(name='baboon'):
    return imread('{}{}'.format(data_dir,_image[name]), mode='F')
