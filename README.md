# Learning to Classify Text Using a Naïve Bayes Classifier

The purpose of this project is to apply a known supervised classification
technique for Natural Language Processing problems. Specifically,
the supervised classification technique used for this problem will be a 
Naïve Bayes Classifier

### Installing

Use these commands to install the software onto the operating system

```
sudo apt-get update
sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy 
    python3-scipy python3-pip libatlas-dev libatlas3gf-base
sudo pip3 install -U scikit-learn
sudo pip3 install nltk

```

Open a python shell and download

```
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('tagsets')

```