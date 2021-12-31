# Import Necessary libraries
import setuptools
# Read README contents
with open("README.md") as fh:
    long_description = fh.read()
# Set the setup file
setuptools.setup(
    name="pysid",
    version="0.0.2",
    author="Eduardo Mapurunga",
    author_email="edumapurunga@gmail.com",
    description="System Identification tools for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edumapurunga/pysid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.19',
                      'scipy>=1.4'],
    python_requires='>=3.6',
)
