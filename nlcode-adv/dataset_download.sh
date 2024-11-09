unzip dataset.zip
cd dataset
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip
python preprocess.py
rm -r python
rm -r *.pkl
rm python.zip
cd ..
