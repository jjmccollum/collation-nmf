from setuptools import setup, find_packages

setup(
    name='collation-nmf',
    version='1.0.0',
    packages=find_packages(),
    description='General-purpose Python tools to perform non-negative matrix factorization on textual collations',
    author='Joey McCollum',
    license='MIT',
    author_email='jjmccollum@vt.edu',
    url='https://github.com/jjmccollum/collation-nmf',
    python_requires='>=3.5',
    install_requires=[
        'pandas',
        'sklearn',
        'scipy',
        'nimfa'
    ],
	classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
