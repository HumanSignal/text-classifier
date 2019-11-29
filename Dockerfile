FROM python:3.7
RUN pip install uwsgi
RUN pip install supervisor
ADD . /app
WORKDIR /app
EXPOSE 9090
RUN pip install -r requirements.txt
CMD ["supervisord", "-c", "supervisord.conf"]