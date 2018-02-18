from setuptools import setup

setup(name="bird_call",
      version='0.1',
      description='tools for anayzing bird vocalizations for signal\
      processing, isolating syllables and reintegrating them into more \
      complex calls.',
      url="https://github.com/jpcurrea/bird_call.git",
      author='Pablo Currea',
      author_email='johnpaulcurrea@gmail.com',
      license='MIT',
      packages=['bird_call'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'pandas',
          'pygame'
      ],
      zip_safe=False)
