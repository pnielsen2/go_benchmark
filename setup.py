from setuptools import setup, find_packages

setup(
    name="go_benchmark",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'pyro-ppl',
        'numpy',
        'alpharedmond'
    ],
    # Optional
    author="Philip Nielsen",
    author_email="pnielsen2@outlook.com",
    description="Go UQ benchmarking and dataset generator",
    license="MIT",
)
