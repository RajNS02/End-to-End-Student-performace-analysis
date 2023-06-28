from setuptools import find_packages, setup
from typing import List

HYPEN_EDOT_FOR_SETUP = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
    Returns the list of requirements
    """
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_EDOT_FOR_SETUP in requirements:
            requirements.remove(HYPEN_EDOT_FOR_SETUP)
    
    return requirements

setup(
name = 'End2End_Student_Performace_Analysis',
version = '0.0.1',
author = 'RajNS02',
author_email = 'rajns0000@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)