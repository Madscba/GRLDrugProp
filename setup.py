"""Setup the package."""

from setuptools import find_packages, setup

find_packages

setup(
    name="graph_package",
    packages=find_packages(),
    version="0.1.0",
    license="Apache License, Version 2.0",
    description="A Deep Learning Library for synergy prediction.",
    author="Villads Stokbro, Mads Berggrein and Johannes Reiche",
    author_email="benedek.rozemberczki@gmail.com",
)