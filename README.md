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
### 1. wavefront and intensity
* [Example 1: Plotting wavefront and intensity figure of light fileds](https://miyoshichi.github.io/SimLight/field.html)

### 2. zernike method
* [Example 2: Using zernike method in SimLight to generate an aberrated light field and fit its coefficients](https://miyoshichi.github.io/SimLight/zernike.html)

### 3. vertical intensity plotting
* [Example 3: Plotting vertical intensity](https://miyoshichi.github.io/SimLight/intensity.html)