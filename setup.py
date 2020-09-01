from distutils.core import setup


setup(
    name = 'numpy-linalg',
    packages = ['linalg'],
    version = '0.1',
    license = 'GPLv3',
    description = 'Collection of wrapper classes to make it easier (more readable) to work with NumPy',
    author = 'Henrik A. Christensen',
    author_email = 'sensen1695@hotmail.com',
    url = 'https://github.com/henrikac/numpy-linalg',
    keywords = ['Vector', 'Matrix', 'NumPy'],
    install_requires = [
        'numpy==1.18.1',
    ],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
)

