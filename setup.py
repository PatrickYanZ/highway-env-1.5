# Following PEP 517/518, this file should not not needed and replaced instead by the setup.cfg file and pyproject.toml.
# Unfortunately it is still required py the pip editable mode `pip install -e`
# See https://stackoverflow.com/a/60885212

from setuptools import setup

if __name__ == "__main__":
    setup()
    
    setup(
    name="gym_examples",
    version="0.0.1",
    # install_requires=["gym==0.26.0", "pygame==2.1.0"],
    )