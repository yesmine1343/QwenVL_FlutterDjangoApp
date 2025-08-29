OCR App – Setup Guide

Backend Setup (Django)

Install Django: pip install django

Go to backend folder: cd backend

Run the server: python manage.py runserver

API available at http://127.0.0.1:8000/api/ocrapi/

Frontend Setup (Flutter)

Go to Flutter project: cd your_flutter_project

Run the app: flutter run

Pick an image and select language

Click "Process Image" to get extracted text

Debug console shows detailed logs

"View JSON" button shows raw API response

Expected Response

Extracted text:
Hello World


Troubleshooting

Make sure Django server is running

Images must be readable and under 10MB

Check Flutter debug console for logs