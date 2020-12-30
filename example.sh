unzip model.zip 

echo "moving model..."
mv -f model/bert-based-model bert-based/ \
    && mv -f model/bert-crf-model bert-crf/ \
    && rm -r model
echo "moving model done."



