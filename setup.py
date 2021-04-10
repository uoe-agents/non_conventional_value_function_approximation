from setuptools import setup
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
        return list(reqs)

def find_version():
    with open('rl_vfa/__init__.py') as fp:
        for line in fp:
            match = re.search(r"__version__\s*=\s*'([^']+)'", line)
            if match:
                return match.group(1)
    assert False, 'cannot find version'

install_requires = load_requirements('requirements.txt')

setup(
    name='rl_vfa',
    version=find_version(),
    packages=['rl_vfa'],
    entry_points={
        'console_scripts': [
            'rl_vfa_d = rl_vfa.httpd:main',
        ]
    },
    install_requires=install_requires,
    description='Function Approximation methods in Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/atsiakkas/rl_vfa',
)
