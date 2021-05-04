from setuptools import setup, find_packages
import re

with open('README.md') as fp:
    long_description = fp.read()

def parse_req_line(line):
    line = line.strip()
    if not line or line[0] == '#':
        return None
    return line

def load_requirements(file_name):
    with open(file_name) as fp:
        reqs = filter(None, (parse_req_line(line) for line in fp))
        print(reqs)
        return list(reqs)

install_requires = load_requirements('requirements.txt')

setup(
    name='rl_vfa',
    version="0.1",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=install_requires,
    description='Function Approximation methods in Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/atsiakkas/rl_vfa',
)
