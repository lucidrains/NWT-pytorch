from setuptools import setup, find_packages

setup(
  name = 'nwt-pytorch',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'NWT - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/NWT-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'pytorch',
    'audio to video synthesis'
  ],
  install_requires=[
    'einops>=0.4',
    'torch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
