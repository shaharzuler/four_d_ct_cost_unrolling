from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f)]


setup(
    name='four_d_ct_cost_unrolling',
    version='0.1.0',
    include_package_data=True,
    # packages=find_packages(),
    install_requires=requirements,
    author='Shahar Zuler',
    author_email='shahar.zuler@gmail.com',
    description='A package for 4DCT scene flow based on https://github.com/gallif/_4DCTCostUnrolling',
    url='https://github.com/shaharzuler/four_d_ct_cost_unrolling',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)