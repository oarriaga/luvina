#!/bin/bash
python setup.py register -r pypitest
python setup.py sdist upload -r pypitest
python setup.py register -r pypi
python setup.py sdist upload -r pypi
sudo pip3 install luvina --upgrade
