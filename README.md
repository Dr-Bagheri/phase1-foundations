# Phase 1 – Hello World Deployment

## Tools Used
- Git & GitHub
- Docker
- Flask
- Google Cloud VM

## Steps
1. Built Flask app
2. Dockerized application
3. Pushed image to Docker Hub
4. Deployed to GPU-enabled VM
5. Exposed via port 80

## Commands
```bash
docker build -t hello-flask .
docker run -p 5000:5000 hello-flask
