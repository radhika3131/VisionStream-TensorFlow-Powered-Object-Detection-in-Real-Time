# Sign_language_Detection_Walkthrough
VisionStream is an advanced object detection system built using TensorFlow's Object Detection API. This project demonstrates the full pipeline of training a custom object detection model on your own dataset, fine-tuning it with pre-trained weights, and deploying it for real-time detection using a webcam. The model is capable of identifying multiple sign language gestures in live video streams, making it suitable for applications in surveillance, interactive systems, and automated monitoring.

![no3](https://github.com/user-attachments/assets/e00621d6-1814-4caa-936d-9809b4bc6aa0)


Key features include:

* Custom Model Training: Train and fine-tune an object detection model on any dataset using TensorFlow.
* Real-Time Detection: Implement live object detection with webcam integration for instant feedback.
* Model Exporting: Convert the trained model into a deployable format for use in production environments.
* Extensive Documentation: Detailed guides and scripts for setting up, training, and deploying the model, making it easy for others to replicate or extend the project.
## Steps
### step 1. 
Clone this repository:

### step 2. 
Create a new virtual environment
```
python -m venv vision
```

### step 3.
Activate your virtual environment

```
.\vision\Scripts\activate # Windows 
```

### step 4.
Install dependencies and add a virtual environment to the Python Kernel
```
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=tfod
```

### step 5.
 Collect images using the Notebook 1. [Image Collection.ipynb](https://github.com/radhika3131/sign_language_Detection/blob/main/1.%20Image%20Collection.ipynb) - ensure you change the kernel to the virtual environment as shown below
![](![Screenshot (1)](https://github.com/radhika3131/sign_language_Detection/assets/102825662/39de89c6-b425-4a59-b131-dc3eb1a3e411)
)

## Data Preparation
*  Step 1: **Dataset Collection**
  * Collect images of the sign language gestures you wish to train on (e.g., "I Love You," "No").
  * Organize the dataset into train and test directories.

    \Folder_Name\Tensorflow\workspace\images\train
    \Folder_Name\Tensorflow\workspace\images\test

* Step 2: **Image Annotation**
  * Use LabelImg to annotate the images, generating XML files corresponding to the annotations.
* Step 3: **Convert Annotations to TFRecord**
Use the provided script to convert annotations to the TFRecord format required for training:
```
!python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')} 
!python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')} 
```
## Training the Model

Begin the training process by opening 2. [Training and Detection.ipynb](https://github.com/radhika3131/sign_language_Detection/blob/main/2.%20Training%20and%20Detection.ipynb), this notebook will walk you through installing Tensorflow Object Detection, making detections, saving and exporting your model.

During this process, the Notebook will install Tensorflow Object Detection. You should ideally receive a notification indicating that the API has been installed successfully at Step 8 with the last line stating OK.
![Screenshot (2)](https://github.com/radhika3131/sign_language_Detection/assets/102825662/8ca5bb31-b230-4180-9323-7712583a8824)

* Step 1: **Configure the Model**
     * Update the pipeline. Configured file with the correct paths to your data and adjusted hyperparameters (e.g., batch size, learning rate).
* Step 2: **Train the Model**
    * Run the training script to start training your model:
  ```
  command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
  
  ```
 * Step 3: **Monitor Training with TensorBoard**
     * Visualize the training process and metrics using TensorBoard:
     *  Navigate to the training folder for your trained model e.g.
         cd Tensorlfow/workspace/models/my_ssd_mobnet/train and open Tensorboard with the following command
          ```
           tensorboard --logdir=. 
          ```
For Example :
![realtime3](https://github.com/user-attachments/assets/80a09677-6673-45c6-99ec-01c112ad0dfb)

  * **Alternative: Monitor Training from Command Line** 
     * You can also monitor the training process directly from the command line, when you run the above training command and print, you will have something like this:
       ```
       python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config 
       --num_train_steps=2000
       ```
   
      You can run the above command in the cmd and This command will display the training progress, including loss values and other metrics, in the command prompt.
      Tensorboard will be accessible through your browser

     ![realtime](https://github.com/user-attachments/assets/a05b4330-1643-4bcf-8c29-3ada230493df)

     ## Model Evaluation

     * Step 1: Run the Evaluation script to start the evaluation of  your model: 
     ```
     command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
     ```
     
       
    * Step 2: **Monitor Evaluation with TensorBoard**
    You can optionally evaluate your model inside of Tensorboard. Once the model has been trained and you have run the evaluation command under Step 1
     *  Navigate to the Evaluation folder for your trained model e.g.
         cd Tensorlfow/workspace/models/my_ssd_mobnet/eval and open Tensorboard with the following command
          ```
           tensorboard --logdir=. 
          ```
          
 ![realtime4](https://github.com/user-attachments/assets/c3deeae0-a319-4c3e-a157-dd9417fd9137)

 * **Alternative: Monitor Evaluation from Command Line** 
     * You can also monitor the Evaluation process directly from the command line, when you run the above evaluation command and print, you will have something like this:
       ```
        python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config 
        --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet
       ```
   
      You can run the above command in the cmd and This command will display the training progress, including loss values and other metrics, in the command prompt.
      Tensorboard will be accessible through your browser  and you will be able to see metrics including mAP - mean Average Precision, and Recall.

     ![realtime2](https://github.com/user-attachments/assets/bd0d5655-72bc-4785-ba8f-cb225a839603)

  ## Run Real-Time Detection
   Use your webcam to perform real-time gesture detection with the exported model
   Results
 ## Training and Evaluation Metrics
   Visualize the training and evaluation metrics, such as loss curves and precision/recall graphs
   * Training Loss Curve:
   * Evaluation Metrics:

## Real-Time Detection Outputs
I Love You Gesture Detection:
![ilove2](https://github.com/user-attachments/assets/14606918-a81c-454d-bdb9-0534379a1c6a)

No Gesture Detection:
![no3](https://github.com/user-attachments/assets/0abbbe12-2319-45ea-af9d-05a71509c40a)




