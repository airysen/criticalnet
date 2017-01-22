from setuptools import setup, find_packages


def lib_import(name):
    for lb in name:
        try:
            exec('import {0}'.format(lb))
        except ImportError as e:
            print("\n{0} not found. {1}. \n".format(lb, e))
            exit()

lib_import(['numpy', 'cv2', 'matplotlib', 'scipy'])

setup(name='criticalnet',
      version='0.1',
      description='Critical Net',
      long_description='Python implementation of the Critical Net computing algorytm',
      classifiers=[
          'Development Status :: Alpha',
          'License :: MIT License',
          'Programming Language :: Python :: 3.4',
          'Topic :: Image matching :: Visual mapping :: Bag-of-feature',
      ],
      url='http://github.com/airysen/criticalnet',
      author='Arseniy Kustov',
      author_email='me@airysen.co',
      license='MIT',
      packages = find_packages(),
      install_requires=['scikit-image', 'networkx'
                        ],
      include_package_data=True,

      zip_safe=False)
