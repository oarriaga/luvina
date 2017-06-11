from setuptools import setup
from setuptools import find_packages

setup(name='luvina',
      version='0.0.3',
      description='High-level API for Natural Language Processing in NLTK',
      author='Octavio Arriaga',
      author_email='arriaga.camargo@gmail.com',
      url='https://github.com/oarriaga/luvina',
      license='MIT',
      install_requires=['nltk', 'pyenchant'],
      packages=find_packages()
      )


