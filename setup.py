from setuptools import setup, find_packages

setup(
    name='modular-python-project',
    version='0.1.0',
    author='Luzius Moll',
    author_email='luzius.moll@gmail.com',
    description='A modular Python project for geometry input and finite element analysis.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
    ],
    entry_points={
        'console_scripts': [
            'modular-python-project=main:main',  # Adjust this based on your main function location
        ],
    },
)