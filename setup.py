import os
from setuptools import setup, find_packages

install_requires = ["numpy", "pandas>=1.0", "HTSeq", "pybedtools", "scipy", "pathos", "sklearn"]
tests_require = ["coverage", "nose", "pysam"]
version = "0.3.1"

# Description from README.md
base_dir = os.path.dirname(os.path.abspath(__file__))
long_description = "\n\n".join([open(os.path.join(base_dir, "README.md"), "r").read()])

setup(
    name="inferelator_prior",
    version=version,
    description="Inferelator-Prior Network Generation Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flatironinstitute/inferelator-prior",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=["inferelator_prior", "inferelator_prior.*"]),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="nose.collector",
)
