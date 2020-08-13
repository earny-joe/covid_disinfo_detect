from setuptools import find_packages, setup


def list_reqs(fname='requirements.txt'):
    with open(fname) as f:
        return f.read().splitlines()


setup(
    name='covid_disinfo_detect',
    packages=find_packages(),
    version='0.1.0',
    description='A project the attempts to detect \
    instances of disinformation related to COVID-19 on Twitter.',
    author='Joseph Earnshaw',
    license='MIT',
    install_requires=list_reqs()
)
