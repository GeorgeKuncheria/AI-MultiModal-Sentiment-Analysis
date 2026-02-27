from sagemaker.pytorch import PyTorchModel
import sagemaker


def deploy_enpoint():
    sagemaker.Session()
    role = "arn:aws:iam::471112871413:role/sentiment-analysis-endpoint-role"

    model_uri = "s3://sentiment-analysis-saas-1227/inference/model.tar.gz"

    model= PyTorchModel(
        model_data=model_uri,
        role = role,
        framework_version="2.3.0",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="sentiment-analysis-model"
    )


    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint"
    )



if __name__ == "__main__":
    deploy_enpoint()