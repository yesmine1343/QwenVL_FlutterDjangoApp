from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from independentforarabic import ocr_with_gemini as ocr_with_gemini_arabic
from independent import ocr_with_gemini as ocr_with_gemini_latin
#type python manage.py runserver to run server. required version py>=3.9
@method_decorator(csrf_exempt, name='dispatch')
class MyView(View):
    def post(self, request):
        # Get the uploaded file
        uploaded_file = request.FILES.get('files') or request.FILES.get('file') or request.FILES.get('image')
        if not uploaded_file:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        # Get the language parameter
        language = request.POST.get('language', 'default')
        
        if language== 'arabic':
            result = ocr_with_gemini_arabic(uploaded_file)
        else:
            result = ocr_with_gemini_latin(uploaded_file)
        
        return JsonResponse({"text": result})

