
echo *** Starting pipeline ***
python create_data.py
echo *** Data created ***
python data_preprocessing.py
echo *** Data preprocessed ***
python model_preparation.py ***
echo *** Model prepared and saved ***
echo === Model testing ===
python model_testing.py