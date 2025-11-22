"""
ASGI config for Slide project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
import os
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
import slides.routing  # 你的 WebSocket routing

from whitenoise import WhiteNoise

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Slide.settings")

# 獲取 Django 的 ASGI 應用程式
django_asgi_app = get_asgi_application()

# 使用 WhiteNoise 來服務靜態檔案
application = WhiteNoise(django_asgi_app)

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            slides.routing.websocket_urlpatterns
        )
    ),
})


