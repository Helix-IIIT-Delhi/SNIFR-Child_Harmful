from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="snifr-child-harmful",
    version="1.0.0",
    author="Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Abu Osama Siddiqui, Sarthak Jain, Priyabrata Mallick, Jaya Sai Kiran Patibandla, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma",
    description="SNIFR: Audio-Visual Alignment for Child Harmful Content Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helix-IIIT-Delhi/SNIFR-Child_Harmful",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
