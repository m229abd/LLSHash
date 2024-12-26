# setup.py

from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="llshash-package",  # Replace with your desired package name
    version="0.1.0",
    author="Mostafa Abdolmaleki",
    author_email="m229abd@gmail.com",
    description=(
        "A locality sensitive hashing implementation"
        "optimized for large data processing."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/m229abd/llshash",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=["numpy>=1.18.0", "ray>=1.0.0"],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "twine",
            "wheel",
        ],
    },
    include_package_data=True,
)
