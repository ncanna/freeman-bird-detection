1. Videos are copied from the Freeman Cameratrap H60.zip.
2. In train and test folders, there is one video for presence and absence of birds.
3. YOLO training procedure cannot use video directly, so we need to convert videos to images first.
    * Two annotated frame examples are provided in '''frames''' folder for test run:
        * Labels: ./frames/labels
        * Images: ./frames/train
    * The sample labels are in YOLO format: ```<class> <x_center> <y_center> <width> <height>```. Corresponding yaml file is ```freeman-bird-detection/data/train.yaml```.
    * **Note** that the yaml file is created for Wanying's test run only, and directory paths need to be updated per user's environment.