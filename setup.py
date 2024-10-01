from setuptools import setup, find_packages

setup(
    name='abepitope',
    version='0.1',
    packages=find_packages(),  
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    package_data={
        'abepitope': [
            'models/abepiscore1.0/*.pt',  # Include all .pt files for AntiInterNet-1.0
            'models/abepitarget1.0/*.pt',  # Include all .pt files for AntiScout-1.0
            'models/hmm_antibody_identification/*.hmm' # include hmm models for antibody identification
        ]
    },
    install_requires=[
        # List your package's dependencies here
    ],
    author='Joakim Clifford',
    author_email='cliffordjoakim@gmail.com',
    description='AbEpiTope: Access the accuracy of modelled antibody-antigen interfaces (PDB/CIF of AlphaFold or experimentally solved structures) and select the most likely antibody to bind a given antigen from a pool of candidates (PDB/CIF of AlphaFold or experimentally solved structures)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mnielLab/AbEpiTope-1.0.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
