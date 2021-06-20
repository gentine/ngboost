import os
from setuptools import setup, find_packages
import sys

assert sys.version_info >= (3, 6, 0), "NGBoost requires Python 3.6+"


def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path) as f:
        return f.read()


setup(
    name="ngboost",
    version="0.1",
    author="Gentine",
    author_email="pg2328@columbia.edu",
    description="Library for probabilistic predictions via gradient boosting.",
    long_description=get_long_description(),
    long_description_content_type="ngboost",
    url="https://github.com/gentine/ngboost",
    license="Apache License 2.0",
    python_requires=">=3.6",
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*','examples', '*.examples', '*.examples.*']),
    #+ find_packages(where="./distns"),
    install_requires=[
        "numpy>=1.17.2",
        "scipy>=1.3.1",
        "scikit-learn>=0.21.3",
        "tqdm>=4.36.1",
        "lifelines>=0.22.8",
    ],
    # tests_require=["pytest", "pre-commit", "black"],
)
