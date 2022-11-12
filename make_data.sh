echo "Start making data!"
mkdir data/final_version
python data/make_data_category.py
python data/make_data_topic.py
python data/make_data_sentiment.py
echo "Finished making data!"