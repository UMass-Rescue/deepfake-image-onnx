# deepfake-image-onnx
This repo demonstrates an example of how to run ONNX models. This example is aimed to help guide students in the first assignment in the Spring offering of 596E. Further improves can be made, especially when it comes to removing unnecessary dependencies (but this isn't a requirement in the first assignment).

## Steps to use the deep fake model

### Create virtual environment and install dependencies
Create a new virtual environment using any tool you prefer. I use `pipenv` for this example. You can use `conda` or `virtualenv` as well.

```bash
pipenv shell
```

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Download the model
Download the model from the following link: [DeepFake model](https://drive.google.com/file/d/1xvJrHs5aJuiVw1X0lIlBzWCEn3WzvjN0/view?usp=sharing). Place the model in the root directory of this project.

### Run the Flask-ML server

Run the following command to start the Flask-ML server:

```bash
python model_server_onnx.py
```

### Command line interface

The command line interface can be used to test the model. Run the following command to test the model:

```bash
# image_dir is the directory containing the images
python deepfake_cli.py --input_dir path/to/image_dir --output_dir path/to/output_dir
```

### Download and run RescueBox Desktop from the following link: [Rescue Box Desktop](https://github.com/UMass-Rescue/RescueBox-Desktop/releases)

#### Open the RescueBox Desktop application and register the model
![RescueBox Desktop](images/register_model.png)

#### Run the model
![RescueBox Desktop](images/run_model.png)
![RescueBox Desktop](images/select_inputs.png)

#### View the results
![RescueBox Desktop](images/view_results.png)

### Attribution
The deepfake model project was done by students in the Fall 24 offering of 596E. Their work has been modified to work with ONNX models here. Their repo can be found [here](https://github.com/aravadikesh/DeepFakeDetector/).

The model we're using was originally created by the authors of the following paper:

```bibtex
@InProceedings{Lanzino_2024_CVPR,
    author    = {Lanzino, Romeo and Fontana, Federico and Diko, Anxhelo and Marini, Marco Raoul and Cinque, Luigi},
    title     = {Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3771-3780}
}