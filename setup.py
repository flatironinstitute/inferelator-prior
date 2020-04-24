import os
from setuptools import setup, find_packages

install_requires = ["numpy", "pandas", "HTSeq", "pybedtools", "scipy", "pathos"]
tests_require = ["coverage", "nose", "pysam"]
version = "0.1.1"

# Description from README.md
base_dir = os.path.dirname(os.path.abspath(__file__))
long_description = "\n\n".join([open(os.path.join(base_dir, "README.md"), "r").read()])

setup(
    name="srrTomat0",
    version=version,
    description="SRR Pipelines: Building matrixes from read data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flatironinstitute/srrTomat0",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=["srrTomat0", "srrTomat0.*"]),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="nose.collector",
)
