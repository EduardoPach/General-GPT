dataset-download:
	wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
	wget http://images.cocodataset.org/zips/train2017.zip
	wget http://images.cocodataset.org/zips/val2017.zip
	unzip -qq annotations_trainval2017.zip
	unzip -qq train2017.zip
	unzip -qq val2017.zip
	rm -rf annotations_trainval2017.zip train2017.zip val2017.zip

dataset-encode:
	echo "Encoding 2017 train images of COCO dataset"
	python ./src/embed_images.py --model-name ViT-L/14  --image-dir ./train2017 --output-dir ./coco_embs --batch-size 512
	echo "Encoding 2017 val images of COCO dataset"
	python ./src/embed_images.py --model-name ViT-L/14  --image-dir ./val2017 --output-dir ./coco_embs --batch-size 512

dataset:
	make dataset-download
	make dataset-encode