from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig


def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="s3://sentiment-analysis-saas-1227/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role="arn:aws:iam::471112871413:role/sentiment-analysis-execution-role",
        framework_version="2.3.0",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch_size": 32,
            "epochs":13
        },
        tensorboard_config=tensorboard_config
    )


    # Start the training job
    estimator.fit({
        "training": "s3://sentiment-analysis-saas-1227/dataset/train",
        "validation": "s3://sentiment-analysis-saas-1227/dataset/dev",
        "test": "s3://sentiment-analysis-saas-1227/dataset/test"
    })


if __name__=="__main__":
    start_training()

