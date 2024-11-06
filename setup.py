from setuptools import setup, find_namespace_packages

setup(
    name="FedRecPolicyEval",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
)