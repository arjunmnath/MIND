FROM python:3.12-slim

WORKDIR /wsgi 

COPY wsgi/ /wsgi/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "wsgi.py"]
