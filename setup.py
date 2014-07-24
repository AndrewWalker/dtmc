from setuptools import setup, find_packages

def read(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    contents = open(path).read()
    return contents

setup(
    name         = 'dtmc',
    version      = '0.0.1',
    description  = 'Discrete Time Markov Chain analysis and simulation',
    long_description = read('README.rst'),
    author       = 'Andrew Walker',
    author_email = 'walker.ab@gmail.com',
    packages     = find_packages(),
    url          = "http://github.com/AndrewWalker/dtmc",
    license      = "MIT",
    classifiers  = [
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Mathematics',
    ]
)
