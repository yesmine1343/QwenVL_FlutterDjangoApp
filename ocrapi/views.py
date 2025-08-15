from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from test_ocr import run_ocr

@method_decorator(csrf_exempt, name='dispatch')
class MyView(View):
    def post(self, request):
        # Get the uploaded file
        uploaded_file = request.FILES.get('files') or request.FILES.get('file') or request.FILES.get('image')
        if not uploaded_file:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        # Get the language parameter
        language = request.POST.get('language', 'default')
        
        # Call your existing run_ocr function
        result = run_ocr(uploaded_file, language)
        
        return JsonResponse(result)
