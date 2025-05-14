from setuptools import find_packages, setup
from typing import List

def get_requirement()->List[str]:
    """
    this Function will return List of requirements
    """
    requirement_lst:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            # Read Lines from the file
            lines = file.readlines()
            # Process each Line
            for line in lines:
                requirements = line.strip()
                #Ignore empty lines and -e .
                if requirements and requirements != '-e .':
                    requirement_lst.append(requirements)
    except FileNotFoundError:
        print("Requirements.txt file not found")
    return requirement_lst
    
    
setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Mohsin",
    author_email="mmohsinbhatti66@gmail.com",
    packages=find_packages(),
    install_require = get_requirement()
    
)