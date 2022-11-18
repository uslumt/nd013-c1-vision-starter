jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root

python -m tensorboard.main --logdir experiments/reference/

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

train: 
python experiments/model_main_tf2.py --model_dir=experiments/reference/experiment0/ --pipeline_config_path=experiments/reference/experiment0/pipeline_new.config


Validation:
python experiments/model_main_tf2.py --model_dir=experiments/reference/experiment0/ --pipeline_config_path=experiments/reference/experiment0/pipeline_new.config --checkpoint_dir=experiments/reference/experiment0/


inference Video:
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif


python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/experiment1/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/


