# akabat_app/middleware.py
import uuid

class AssignUUIDMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.COOKIES.get("user_uuid"):
            request.new_user_uuid = str(uuid.uuid4())
        response = self.get_response(request)
        if hasattr(request, "new_user_uuid"):
            response.set_cookie("user_uuid", request.new_user_uuid, max_age=60 * 60 * 24 * 365)
        return response
