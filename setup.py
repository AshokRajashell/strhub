from setuptools import setup, find_packages
import codecs
import os



VERSION = '1.0.7'
DESCRIPTION = 'strhub'

# Setting up
setup(
    name="strhub",
    version=VERSION,
    author="Ashok Raja",
    author_email="A.Palaniswamy@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={"strhub":["configs/*.yaml","configs/model/*","models/abinet/*","models/crnn/*","models/parseq/*",
                           "models/trba/*","models/vitstr/*","models/*.py"]},
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
