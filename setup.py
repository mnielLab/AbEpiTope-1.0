from setuptools import setup, find_packages

setup(
    name='your-package-name',
    version='0.1',
    packages=find_packages(where='src'),  # Look for packages in the 'src' directory
    package_dir={'': 'src'},  # Set package root to 'src'
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={
        'your_package': [
            'models/modeltype1/*.pt',  # Include all .pt files for modeltype1
            'models/modeltype2/*.pt',  # Include all .pt files for modeltype2
        ]
    },
    install_requires=[
        # List your package's dependencies here
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-repo',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
