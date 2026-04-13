from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "lcm.lcm",
        ["src/lcm/lcm.pyx"],
        include_dirs=[np.get_include()],
        language="c",
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "lcm.base",
        ["src/lcm/base.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "lcm.queue",
        ["src/lcm/queue.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "lcm.itemset",
        ["src/lcm/itemset.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "lcm.trsact",
        ["src/lcm/trsact.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "lcm.problem",
        ["src/lcm/problem.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "cfptree.main",
        ["src/cfptree/cfptree.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
    Extension(
        "cfptree.item",
        ["src/cfptree/item.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    ),
]

setup(
    name="fcptree-lcm-itemset",
    version="1.0.0",
    description="Cython port of Linear time Closed itemset Miner and CFP-tree structure",
    author="Aryanto",
    url='https://github.com/masterofray/cooprecsys_Association_Rule_Learning',
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=[
        "numpy>=1.19.0",
        "Cython>=0.29.0",
    ],
    python_requires=">=3.7",
    include_dirs=[np.get_include()],
    install_requires=[
        'Cython>=0.29.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'networkx>=2.5',
        'pandas>=1.1.0',
        'seaborn>=0.11.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Cython',
    ],
)