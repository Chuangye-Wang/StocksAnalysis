import setuptools
#from stock_analysis import __version__ as version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stock_analysis", # Replace with your own username
    version='Python 3.7.6',
    keywords=['stock analysis techniques', 'investment', 'python'],
    install_requires=['numpy', 'matplotlib', 'pandas', 'pandas_datareader', 'datetime'],
    author="Chuangye Wang",
    author_email="ustcwcy@gmail.com",
    description="A python library for technique analysis of stocks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/Chuangye-Wang/StocksAnalysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'Topic :: Finance :: stocks',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
