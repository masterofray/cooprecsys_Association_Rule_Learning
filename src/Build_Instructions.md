# LCM Python/Cython Port - Build Instructions

## Quick Start

1. Extract the ZIP file
2. Install dependencies:
   pip install -r requirements.txt

3. Build Cython extensions:
   python setup.py build_ext --inplace

4. Install the package:
   pip install -e .

5. Try the examples:
   python examples/example_usage.py

## Installation Steps

### Option A: Development Installation
python setup.py develop

### Option B: Standard Installation
python setup.py install

### Option C: Pip Installation
pip install .

## Testing

Run the test suite:
python -m pytest tests/

Run examples:
python examples/example_usage.py

## Troubleshooting

If you encounter build errors:
1. Ensure you have a C compiler installed
   - Windows: Visual Studio C++ or MinGW
   - macOS: Xcode Command Line Tools
   - Linux: gcc/clang

2. Update build tools:
   pip install --upgrade setuptools Cython

3. Clean build directory:
   python setup.py clean --all
   rm -rf build/ dist/ *.egg-info

4. Rebuild:
   python setup.py build_ext --inplace