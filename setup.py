import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ghj_pkg", # Replace with your own username
    version="0.0.1",
    author="Haojie Guo",
    author_email="ghj_xin@163.com",
    description="This is a mini but function completed neural networks framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghj-hey",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)