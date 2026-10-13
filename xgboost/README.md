# Train, Register, Deploy and Serve XGBoost Classifier to AI Inference Service Programmatically

![alt text](../img/xgboost-deployment.png)

## Objective

In this tutorial you will learn how to build an XGBoost classifier and deploy it to the Cloudera AI Inference Service. First, you will train and register the model with the Cloudera AI Registry; Then, you will create an AI Inference Service Endpoint to serve predictions in real time from your secure endpoint.

If you'd like to use a different framework you can apply the same steps with other Machine Learning (Discriminative AI) models such as SciKitLearn, SparkMllib, PyTorch, etc.

### Motivation

With Cloudera AI, enterprises can train predictive models and securely host them in API Endpoints within their Public or Private Cloud, in order to build AI-powered applications at scale.

### Cloudera AI for Predictive Modeling

Cloudera AI empowers enterprise users to build and deploy predictive models at scale by seamlessly integrating popular frameworks like Apache Spark and PyTorch. With Spark’s distributed processing and PyTorch’s deep learning capabilities, data teams can train complex models efficiently on large datasets. Cloudera also supports model lifecycle management through MLflow Registry, allowing users to track, version, and manage models from experimentation to production within a unified platform. This integration streamlines the machine learning workflow, enhancing collaboration, traceability, and operational efficiency across teams.

### Hybrid Enterprise AI with CAI

Cloudera AI (CAI) is a core component of Cloudera’s hybrid cloud data platform, which is designed to operate seamlessly across both private and public cloud environments. This hybrid architecture allows organizations to deploy AI models securely wherever their data resides—on-premises for sensitive workloads or in the public cloud for greater scalability and flexibility. With Cloudera AI, enterprises can maintain governance, compliance, and control over their machine learning pipelines while taking advantage of cloud-native capabilities. This ensures that large language models and other AI applications can be deployed and managed securely across diverse IT environments without compromising performance or data privacy.

## Requirements

This example was built with Cloudera On Cloud Public Cloud Runtime 7.3.1, CAI Workbench 2.0.50, Inference Service 1.4.0 and AI Registry 1.7.0. You can reproduce this tutorial in your CAI environment with the following:

* A CAI Environment in Private or Public Cloud.
* An AI Registry deployment.
* An AI Inference Service deployment.

## Useful Documentation Links

* How to deploy a Workbench in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/workspaces-privatecloud/topics/ml-pvc-provision-ml-workspace.html
* How to deploy an AI Registry in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/setup-model-registry/topics/ml-setting-up-model-registry.html
* How to deploy an AI Inference Service in Cloudera AI: https://docs.cloudera.com/machine-learning/1.5.5/setup-cloudera-ai-inference/topics/ml-caii-use-caii.html
* How to set up a Hugging Face Account: https://huggingface.co/welcome
* How to set up the CDP CLI: https://docs.cloudera.com/cdp-public-cloud/cloud/cli/topics/mc-cdp-cli.html
