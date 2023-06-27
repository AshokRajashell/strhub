from setuptools import setup, find_packages
import codecs
import os



VERSION = '1.0.4'
DESCRIPTION = 'strhub'

# Setting up
setup(
    name="strhub",
    version=VERSION,
    author="Ashok Raja",
    author_email="A.Palaniswamy@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    package_name={"strhub":["configs/main.yaml","configs/model/abinet.yaml","configs/model/crnn.yaml","configs/model/parseq.yaml","configs/model/trba.yaml","configs/model/vitstr.yaml"]},    
    install_requires=[],
    keywords=['python', 'video', 'stream', 'video stream', 'camera stream', 'sockets'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
