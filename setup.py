from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='t0',
    version='0.0.0',
    url='https://github.com/bigscience-workshop/t-zero.git',
    author='Multiple Authors',
    author_email='xxx',
    python_requires='>=3.7, <3.8', # TODO: update when https://github.com/bigscience-workshop/promptsource/issues/584 is fixed
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    description='Multitask Prompted Training Enables Zero-Shot Task Generalization',
    packages=find_packages(),
    license="Apache Software License 2.0",
    long_description=readme,
    install_requires=[
        "promptsource",
        "accelerate",
        "transformers",
        "torch",
        "datasets",
        "jinja2",
        "datasets",
        "sentencepiece",
        "protobuf",
        "scikit-learn"
    ],
    extras_require={
        "seqio_tasks": [
            "seqio",
            "t5",
            "tensorflow",
        ]
    },
    package_data={
        "": [
            "datasets.csv",
        ]
    }
)
