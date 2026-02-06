FROM python:3.8
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN mkdir /main/
RUN chmod -R 777 /main/
ADD connect4_backend.py /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./connect4_backend.py" ]