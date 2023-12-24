pip install -q kaggle
kaggle datasets download -d awsaf49/asvpoof-2019-dataset
unzip -q ./asvpoof-2019-dataset.zip
rm asvpoof-2019-dataset.zip
rm -r PA
