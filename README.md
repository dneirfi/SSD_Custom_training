# SSD_Custom_training
ssd custom training using Tensorflow object detection api
## 자동화 방법 연구중

1. 환경
OS : ubuntu 16.04
python version : 3.5.2
CUDA version : 9.0
cuDNN version : 7.0.5
tensorflow version : 1.12.0
tensorboard : 1.12.0

2. Train ( SSD_Train.pptx)
###필요 패키지 설치 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

cd SSD_custom_training

### tensorflow, tensorboard 설치 
sudo pip install tensorflow-gpu==1.12.0
sudo pip install tensorboard==1.12.0
git clone https://github.com/tensorflow/models# Clone Tensorflow models



### coco api 설치(optional)
cd SSD_custom_training
git clone https://github.com/cocodataset/cocoapi.git
 
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
( prococotools는 Object Detection 모델을  evaluation할 때 사용하는 evaluation metrics로 사용된다. 이 후 coco evaluation metrics를 사용하지 않더라도, tensorflow object detection api 는 내부적으로 coco evaluation metrics를 기본으로 사용하기때문에 설치)

### Protobuf 컴파일(models/research) 
 ( /models/research/)
protoc object_detection/protos/*.proto --python_out=.
(((Note: If you're getting errors while compiling, you might be using an incompatible protobuf compiler. If that's the case, use the following manual installation
Download and install the 3.0 release of protoc, then unzip the file.

### From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
Run the compilation process again, but use the downloaded version of protoc

### From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
)))

###PYTHONPATH 라이브러리 추가 (.bashrc에 추가 안할 경우 새 터미널 창마다 PYTHONPATH 라이브러리 추가해주어야함) 
(/models/research/)
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
Tensorflow Object Detection API 가 python 3.x 버전에서 많은 오류 발생 → tensorflow/models github repo 의 issues에서 오류찾아 해결가능

###입력 데이터 준비
SSD_Train_방법.pptx 의 "Training Dataset 준비하기" 참고
(models/research/object_detection/ )

mkdir images
cd images
mkdir test
mkdir train
학습시키고자 하는 이미지들을 images 폴더에 넣고 9:1의 비율로 train 과 test 폴더에도 복사한다
LabelImage 이용(models/research/object_detection/ )

git clone https://github.com/tzutalin/labelImg
w키로 바운딩박스 만들고 ctrl+s 로 저장하면 xml파일이 생성된다. 모든 이미지에 대해 반복한다.



Xml 파일을 TFRecord파일로 변환.(models/research/object_detection/ )

git clone https://github.com/datitran/raccoon_dataset
*.xml파일들의 데이터를 하나의 csv파일로 변환하기 위해 xml_to_csv.py스크립트를 수정(models/research/object_detection/ )



python3 xml_to_csv.py

결과 




csv파일을 TFRecord파일로 변환하기 위해  generate_tfrecord.py 스크립트를 수정(models/research/object_detection/ )



python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
결과 
(위 사진의 경로 무시 - 잘못되었음) 

사전 학습된 모델 다운로드 후 압축풀기 
(models/research/object_detection/ )

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
(models/research/object_detection/samples/configs/)의 ssd_mobilenet_v1_pets.config 를 다음과 같이 수정 후 object_detection/training 폴더를만들고 그 안에 copy and paste

위 사진의 num_classes 수 알맞게 조정
(object_detection/) data 디렉토리 안에 object-detection.pbtxt 파일 추가 및 수정


Training
(models/research/object_detection/) 

python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
(models/research/object_detection/) (텐서보드로 loss 값 확인가능)

tensorboard --logdir=./training
Exporting the Tensorflow Graph
(models/research/object_detection/) 

python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path training/ssd_mobilenet_v1_pets.config \
--trained_checkpoint_prefix training/model.ckpt-{CHECKPOINT_NUMBER} \
--output_directory inference_graph_trafficlight_car

(trained_checkpoint_prefix 는 checkpoint_number 확인 후 알맞게 수정, output_directory 이름은 학습시킬 class에 맞게 수정)

Test
(models/research/object_detection/test_images/) 테스트 하고자 하는 이미지들을 image{number}.jpg 로 저장
jupyter notebook 실행 후 (object_detection/)object_detection_tutorial.ipynb 실행 하면 test_images 의 이미지들 실행 결과 확인 가능



번외 
웹캠이용한 실시간 사물 인식 테스트 https://towardsdatascience.com/real-time-object-detection-with-tensorflow-detection-model-e7fd20421d5d (jupyter notebook version)
https://github.com/ukayzm/opencv/tree/master/object_detection_tensorflow (python code version)

에러
import object_detection 에러 : 진행 디렉토리 다시 확인해보기. models/research/object_detection 
cuda 9.0 에러 : tensorflow 모델 downgrade 필요.
no module named 'nets' : PYTHONPATH 추가 다시하기.  (models/research/) export PYTHONPATH="$PYTHONPATH:/home/username/work_directory/models/research/slim"
