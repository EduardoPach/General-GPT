build-dataset:
	wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
	wget http://images.cocodataset.org/zips/train2017.zip
	unzip -qq annotations_trainval2017.zip
	unzip -qq train2017.zip
	rm -rf annotations_trainval2017.zip train2017.zip