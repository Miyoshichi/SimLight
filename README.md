# SimLight
A toolkit helps optical simulation.

## Installation
### install from pypi
```
$ pip install SimLight
```
### or you can download from github
```
$ git clone https://github.com/Miyoshichi/SimLight.git
$ python3 install setup.py
```

## How to use
### basic usage
```python
import SimLight as sl
```
### you can also import subpackages like this
```python
import SimLight.zernike as slz
import SimLight.plottools as slpl
```

## Examples
### 1. zernike method
* [Example 1: Using zernike method in SimLight to generate an aberrated light field and fit its coefficients](https://miyoshichi.github.io/SimLight/zernike.html)

### 2. vertical intensity plotting
* [Example 2: Plotting vertical intensity](https://miyoshichi.github.io/SimLight/intensity.html)