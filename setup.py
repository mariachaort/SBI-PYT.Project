from setuptools import setup, find_packages

setup(
    name='protein_binding_predictor',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'protein_binding_predictor=main:main',
            'model_generator=model_generator:main',
            'evaluate_model=evaluate_model:main'
        ],
    },
    install_requires=[
        'biopython',
        'scikit-learn',
        'numpy',
        'pandas',
        'joblib',
        'matplotlib',
        'seaborn'
    ],
    author='Chacón M., Delgado P., Ascunce A.',
    description='Protein Binding Site Predictor: A tool for predicting protein binding sites using machine learning.',
    keywords='protein binding site predictor machine learning',
    url='https://github.com/mariachaort/SBI-PYT.Project.git',
)

