# install all NLTK models https://www.nltk.org/data.html
echo "installing nltk data sets, corpora and models"
# TODO this could be optimized. Not everything needs to be downloaded
python -m nltk.downloader all 