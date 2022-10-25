import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="schooled",
    version="0.0.3",
    description="Stuff",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/schooled",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["schooled","schooled.cnvrg","schooled.datasets"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["wheel","timemachines","cnvrg2"],
    entry_points={
        "console_scripts": [
            "schooled=schooled.__main__:main",
        ]
    },
)
