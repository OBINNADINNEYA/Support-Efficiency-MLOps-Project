from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)-> List[str]:
    '''
    this function will return thr list of requirments
    '''

    requirements = []

    with open(file_path,'r') as file:
        for line in file:
            requirements.append(line.rstrip('\n'))

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(

    name = 'support_efficieny_mlops_v1',
    version='0.0.1',
    author='Obinna',
    author_email='obinnadinneya@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

)