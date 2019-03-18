import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tools-and-scripts",
    version="0.0.1",
    author="Kevin Sebesta",
    author_email="kevin.w.sebesta@gmail.com",
    description="Tools and scripts for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kwsebesta/tools-and-scripts",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
