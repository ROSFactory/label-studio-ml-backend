# Quickstart

0. Update the environment variables in docked-compose.yml
1. Build and start Label Studio and Machine Learning backend on `http://ml-backend:9090`

```bash
docker-compose up --build
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/health
{"status":"UP"}
```

3. Connect to the backend from Label Studio: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://ml-backend:9090` as a URL.