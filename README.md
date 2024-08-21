# Sign_language_Detection_Walkthrough
TensorVision is an advanced object detection system built using TensorFlow's Object Detection API. This project demonstrates the full pipeline of training a custom object detection model on your own dataset, fine-tuning it with pre-trained weights, and deploying it for real-time detection using a webcam. The model is capable of identifying multiple objects in live video streams, making it suitable for applications in surveillance, interactive systems, and automated monitoring.

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
python -m venv tfod
```

### step 3.
Activate your virtual environment

```
.\tfod\Scripts\activate # Windows 
```

### step 4.
Install dependencies and add a virtual environment to the Python Kernel
```
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=tfod
```

### step 5.
Step 5. Collect images using the Notebook 1. [Image Collection.ipynb](https://github.com/radhika3131/sign_language_Detection/blob/main/1.%20Image%20Collection.ipynb) - ensure you change the kernel to the virtual environment as shown below
![](![Screenshot (1)](https://github.com/radhika3131/sign_language_Detection/assets/102825662/39de89c6-b425-4a59-b131-dc3eb1a3e411)
)

### step 6.
Manually divide collected images into two folders train and test. So now all folders and annotations should be split between the following two folders.
\Folder_Name\Tensorflow\workspace\images\train
\Folder_Name\Tensorflow\workspace\images\test

### step 7. 
Begin training process by opening 2. [Training and Detection.ipynb](https://github.com/radhika3131/sign_language_Detection/blob/main/2.%20Training%20and%20Detection.ipynb), this notebook will walk you through installing Tensorflow Object Detection, making detections, saving and exporting your model.

### step 8.
During this process, the Notebook will install Tensorflow Object Detection. You should ideally receive a notification indicating that the API has been installed successfully at Step 8 with the last line stating OK.
![Screenshot (2)](https://github.com/radhika3131/sign_language_Detection/assets/102825662/8ca5bb31-b230-4180-9323-7712583a8824)

### Step 9. 
Once you get to step 6. Train the model, inside of the notebook, you may choose to train the model from within the notebook. I have noticed however that training inside of a separate terminal on a Windows machine you're able to display live loss metrics.

### Step 10. 
You can optionally evaluate your model inside of Tensorboard. Once the model has been trained and you have run the evaluation command under Step 7. Navigate to the evaluation folder for your trained model e.g.
 cd Tensorlfow/workspace/models/my_ssd_mobnet/eval
and open Tensorboard with the following command
tensorboard --logdir=. 
Tensorboard will be accessible through your browser and you will be able to see metrics including mAP - mean Average Precision, and Recall.
