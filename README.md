# Cloud Challenge

### Objective 
- Develop a machine learning model for the Titanic dataset.  
- Implement tests for the model.  
- Use MLflow for performance tracking.  
- Containerize the model with Docker.  
- Automate Docker build and push via GitHub Actions, including test execution.  
- Set up a DAG pipeline using pub-sub messaging to manage and track container execution sequentially.  

## Docker commands

```
docker build -t cloud-challenge .
docker run --rm -v "C:\Users\Sharat\Documents\cloud-challenge\data:/app/data" -v "C:\Users\Sharat\Documents\cloud-challenge\mlruns:/app/mlruns" cloud-challenge  --json_args='{\"steps\": [\"src/data_prep.py\", \"src/train.py\", \"src/predict.py\"]}'
docker tag cloud-challenge:latest sharatsachme/cloud-challenge:latest
docker push sharatsachme/cloud-challenge:latest
```