pip install -q kaggle
kaggle datasets download -d awsaf49/asvpoof-2019-dataset
unzip -q ./asvpoof-2019-dataset.zip
rm asvpoof-2019-dataset.zip
rm -r PA
rm LICENSE_text.txt
rm README.txt
rm asvspoof2019_Interspeech2019_submission.pdf
rm asvspoof2019_evaluation_plan.pdf
