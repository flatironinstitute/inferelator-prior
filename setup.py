from setuptools import setup

install_requires = ["numpy", "pandas"]
tests_require = ["coverage", "nose"]
version = "0.1"

setup(
    name = "srrTomat0",
    version = version,
    description = "SRR Pipelines: Building matrixes from raw read data",
    url = "https://github.com/cskokgibbs/srrTomat0",
    author = "Claudia Skok Gibbs",
    author_email = "cskokgibbs@flatironinstitute.org",
    maintainer = "Claudia Skok Gibbs",
    maintainer_email = "cskokgibbs@flatironinstitute.org",
    packages = ["srrTomat0"],
    zip_safe = False,
    install_requires = install_requires,
    tests_require = tests_require,
    test_suite = "nose.collector",
)
