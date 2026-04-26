# 🎬 Multimodal AI Video Sentiment Model

> A deep learning model built with PyTorch that accepts video input and predicts **sentiment** and **emotion** using multimodal fusion of video, audio, and text signals.

---

## ✨ Features

| Category | Feature |
|----------|---------|
| 🎥 **Video** | Video sentiment analysis & frame extraction |
| 🎙️ **Audio** | Audio feature extraction |
| 📝 **Text** | Text embedding with BERT |
| 🔗 **Fusion** | Multimodal fusion architecture |
| 📊 **Classification** | Emotion and sentiment classification |
| 🚀 **Training** | Model training, evaluation & TensorBoard logging |
| ☁️ **Cloud** | AWS S3 storage & SageMaker endpoint integration |
| 🔐 **SaaS** | User auth, API key management & usage quota tracking |
| 🎨 **UI** | Modern UI with Next.js, React, Tailwind CSS & Auth.js |

---

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Andreaswt/ai-video-sentiment-model.git
cd ai-video-sentiment-model
```

### 2. Install Python

Download and install Python if not already installed → [Python Download](https://www.python.org/downloads/)

### 3. Install Dependencies

```bash
pip install -r training/requirements.txt
```

### 4. Download the Dataset

Visit the [MELD Dataset](https://affective-meld.github.io/) page to download the dataset.

Extract it and place the contents in the `dataset/` directory.

> 📌 **Tip:** Check out the [Emotion Recognition Benchmark](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld) for state-of-the-art model comparisons on the MELD dataset.

---

## 🚀 Training on AWS SageMaker

### Step 1 — Request a Quota Increase

Request a quota increase for a SageMaker training instance (e.g. `ml.g5.xlarge`) in your AWS account.

### Step 2 — Upload Dataset to S3

Upload the MELD dataset to an S3 bucket of your choice.

### Step 3 — Create an IAM Role

Attach the following policies to your role:

- `AmazonSageMakerFullAccess`
- Custom S3 access policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "VisualEditor0",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

### Step 4 — Start the Training Job

```bash
python train_sagemaker.py
```

---

## 📦 Deploying the Endpoint

### Step 1 — Create a Deployment IAM Role

Attach the following policies:

- `AmazonSageMakerFullAccess`
- `CloudWatchLogsFullAccess`
- Custom S3 access policy (same as above)

### Step 2 — Upload Model to S3

Place your trained model file (`.tar.gz`) in your S3 bucket.

### Step 3 — Deploy the Endpoint

```bash
python deployment/deploy_endpoint.py
```

---

## 🔑 Invoking the Endpoint

### Step 1 — Create an IAM User

Attach the following inline policy to your IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": ["arn:aws:s3:::sentiment-analysis-saas/inference/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["sagemaker:InvokeEndpoint"],
      "Resource": [
        "arn:aws:sagemaker:us-east-1:784061079855:endpoint/sentiment-analysis-endpoint"
      ]
    }
  ]
}
```

### Step 2 — Invoke from JavaScript

Use the IAM user credentials with your preferred AWS SDK or NPM library to invoke the endpoint from your Next.js app.

---

## 📈 Accessing TensorBoard

```bash
# Download logs from S3 to your local machine
aws s3 sync s3://your-bucket-name/tensorboard ./tensorboard_logs

# Start the TensorBoard server
tensorboard --logdir tensorboard_logs
```

Then open your browser and visit: [http://localhost:6006](http://localhost:6006)

---

## 🧰 Tech Stack

**Model & Training**
- PyTorch — model architecture & training loop
- BERT — text feature embeddings
- AWS SageMaker — managed training jobs & endpoints
- AWS S3 — dataset and model artifact storage
- TensorBoard — training metrics visualization

**SaaS Application**
- Next.js + React — frontend framework
- Tailwind CSS — styling
- Auth.js — user authentication
- T3 Stack — full-stack foundation

---

