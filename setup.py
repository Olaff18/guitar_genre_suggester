from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'nam_backend',
        ['nam_backend.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    )
]

setup(
    name='nam_backend',
    ext_modules=ext_modules,
)
