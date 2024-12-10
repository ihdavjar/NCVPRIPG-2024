# **Description**: This is the README file for the project.

- `models/`: This directory contains all the two models, which are a part of this submission. Detailed Description of each of the model will be included in the write-up which would be sent via email (`raj.31@iitj.ac.in`) before the given deadline.
- `src/`: This directory contains all the model declarations and the code for inference along with the required preprocessing and post-processing steps.
- `env.yml`: Environment file to replicate the repository.
  Create and activate a virtual environment using `conda`(Recommended).
- `submission.txt`: Inference results for the shared test data (**SCSPAR24_Testdata**) in similar format to (train.txt).

## Below are the steps to create a virtual environment:

1) Open terminal.
2) Change the directory to the location of the `env.yml` file.
3) Utilise the below command to create the virtual environment.

```
conda env create -f env.yml
```

4) Activate the environment with the below command.

```
conda activate SCSPAR24
```

## Follow the below instructions for the **inference** purpose:

1) Open terminal.
2) Change the directory to the `src/`.
3) Use the below command to use the inference pipeline.

```
python inference.py --folder path_to_image_folder --model 1
```

4) Model argument can be `1` and `2` for utilising respective models.
5) Below is an example of the inference command, which may differ in your environment based on the path of the images folder.

```
python inference.py --folder /mnt1/Research/Competitions/Vehant/Submission/SCSPAR24_Testdata --model 1
```

**Note:** Make sure that the images are given integer names as the inference pipelines sorts them in increasing order assuming integer. Which may fail if the names are not intger.

## Resources Used:

1) GPU -> Nvidia RTX 3060
2) CPU -> AMD Ryzen 5 3600x
3) RAM -> 16GB

These above resources were used for training purpose/inference purpose. However the inference was also tested on low end CPU like (Intel i3 5th gen), the inference speed were still reasonably good.

Inference speed can be increased by utilising batched inferencing which is not implemented here.