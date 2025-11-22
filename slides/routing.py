from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/speech/(?P<pdf_id>\w+)/$', consumers.SpeechConsumer.as_asgi()),]
