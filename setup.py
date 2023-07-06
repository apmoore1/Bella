from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='bella_tdsa',
      version='0.3.29',
      description='Target Dependent Sentiment Analysis (TDSA) framework.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/apmoore1/Bella',
      author='Andrew Moore',
      author_email='andrew.p.moore94@gmail.com',
      license='MIT',
      install_requires=[
          'filelock>=3.0.10',
          'Keras>=2.1.3',
          'tensorflow>=1.3.0',
          'scipy==1.10.0',
          'scikit-learn==0.19.1',
          'gensim>=3.0.1',
          'requests>=2.18.4',
          'tqdm>=4.23.4',
          'twokenize>=1.0.0',
          'nltk>=3.2.5',
          'stanfordcorenlp==3.7.0.2',
          'ftfy>=5.2.0',
          'ruamel.yaml>=0.15.34',
          'pandas>=0.21.0',
          'networkx>=2.0',
          'tweebo-parser-python-api>=1.0.4',
          'seaborn>=0.8.1',
          'spacy>=2.0.18,<2.2',
          'graphviz>=0.8.4',
          'pydot>=1.2.4'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.6',
          'Topic :: Text Processing',
          'Topic :: Text Processing :: Linguistic',
      ])
