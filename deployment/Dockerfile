FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements_deploy.txt

EXPOSE 5000
CMD ["python", "deployment/app.py"]
