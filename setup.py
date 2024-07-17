from setuptools import setup
setup(
name="stvelo",
    version="0.1.0",
    description="Package to perform spatial rna velociy",
    author="Sergio Marco Salas",
    author_email="sergiomarco.salas@scilifelab.se",
    packages=["stvelo"],
    install_requires=["scvelo","scanpy"
        # List your module's dependencies here
    ])