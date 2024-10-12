from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import urllib.request
import zipfile
import shutil
import os

def setup_environment():
    
    print("Setting up environment...")
    directories = ['results', 'models', 'plots', 'data', 'log', 'data/insectsound']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    if os.path.exists(os.path.join('data/insectsound', 'InsectSound_TRAIN.arff')):
        print("InsectSound dataset already exists. Skipping download.")
        return

    print("Downloading the InsectSound dataset...")
    url = "https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip"
    zip_path = os.path.join('data', 'InsectSound.zip')
    
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    os.remove(zip_path)

    # move the arff files to the correct folder
    for filename in os.listdir('data'):
        shutil.move(os.path.join('data', filename), os.path.join('data', 'insectsound', filename))
    
    # remove the empty folder
    shutil.rmtree('data/InsectSound')
    print("InsectSound dataset downloaded and extracted successfully.")


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        setup_environment()

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        setup_environment()

setup(
    name='saliencyserieslab',
    description='A package for saliency based explanation methods for time series classification',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Eirik Matias Vinje',
    author_email='eirik.matias@gmail.com',
    url='https://github.com/EirikVinje/saliencyserieslab',
    packages=[
        "saliencyserieslab",
        ],
    install_requires=[        
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'torch',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)