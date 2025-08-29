import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qwen OCR App',
      theme: ThemeData(
        fontFamily: 'Quicksand',
        textTheme: const TextTheme(bodyMedium: TextStyle(color: Colors.black)),
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal.shade200),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  bool _isProcessing = false;
  String? _extractedText;
  double? _confidence;
  String? _errorMessage;
  String? _selectedLanguage;
  String? _rawJsonResponse;
  
  // Store file data instead of File object for web compatibility
  Uint8List? _selectedFileBytes;
  String? _selectedFileName;

  // API Configuration - URLs for HTTP requests to Django backend
  static const String _baseUrl = 'http://127.0.0.1:8000';
  static const String _apiEndpoint = '/api/ocrapi/';

  // Fixed file picker function that works on both web and mobile
  Future<void> _pickImage(String language) async {
    try {
      // Open file picker dialog to select image file
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        allowMultiple: false,
        dialogTitle: 'Select image for $language processing',
        withData: true, // Important: This ensures bytes are loaded on web
      );

      if (result != null && result.files.isNotEmpty) {
        PlatformFile file = result.files.first;
        
        // Get file bytes (works on both web and mobile)
        Uint8List? fileBytes = file.bytes;
        
        if (fileBytes != null) {
          // Store selected file data and language for HTTP request
          setState(() {
            _selectedFileBytes = fileBytes;
            _selectedFileName = file.name;
            _selectedLanguage = language;
            _extractedText = null;
            _confidence = null;
            _errorMessage = null;
          });

          if (kDebugMode) {
            debugPrint('Selected image: ${file.name} (${fileBytes.length} bytes) for $language');
          }

          // Automatically trigger HTTP POST request to OCR endpoint
          await _processImage();
        } else {
          // Fallback for mobile platforms where bytes might be null
          if (!kIsWeb && file.path != null) {
            File imageFile = File(file.path!);
            fileBytes = await imageFile.readAsBytes();
            
            setState(() {
              _selectedFileBytes = fileBytes;
              _selectedFileName = file.name;
              _selectedLanguage = language;
              _extractedText = null;
              _confidence = null;
              _errorMessage = null;
            });

            await _processImage();
          } else {
            _showError('Failed to load image data');
          }
        }
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('Error picking file: $e');
      }
      _showError('Error selecting image: $e');
    }
  }

  // Main HTTP request function - sends image file and language to OCR API endpoint
  // This function creates a multipart POST request to the Django backend
  Future<void> _processImage() async {
    // Validate that we have both file data and language before making HTTP request
    if (_selectedFileBytes == null || _selectedLanguage == null) {
      _showError('Please select an image and language first');
      return;
    }

    // Set processing state to show loading indicator during HTTP request
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      // Create multipart POST request to OCR API endpoint
      // MultipartRequest is used because we need to send both file and form data
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl$_apiEndpoint'),
      );

      // Create multipart file from bytes (works on both web and mobile)
      var multipartFile = http.MultipartFile.fromBytes(
        'file', // Field name expected by Django backend
        _selectedFileBytes!,
        filename: _selectedFileName ?? 'image.jpg',
      );
      request.files.add(multipartFile);

      // Add language parameter to form data - backend expects 'language' field
      // Valid values: 'arabic', 'french', 'english', 'default'
      request.fields['language'] = _selectedLanguage!;

      // Log HTTP request details for debugging
      if (kDebugMode) {
        debugPrint('Sending request to: $_baseUrl$_apiEndpoint');
        debugPrint('Language: $_selectedLanguage');
        debugPrint('File: $_selectedFileName (${_selectedFileBytes!.length} bytes)');
      }

      // Execute HTTP POST request and wait for Django backend response
      var response = await request.send();

      // Parse JSON response from Django backend
      var responseData = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseData);

      // Log complete HTTP response from Django backend for debugging
      if (kDebugMode) {
        debugPrint('=== OCR API RESPONSE ===');
        debugPrint('Response status: ${response.statusCode}');
        debugPrint('Response headers: ${response.headers}');
        debugPrint('Response data: $jsonResponse');
        debugPrint('========================');
      }

      // Handle successful HTTP response (status 200) from OCR endpoint
      if (response.statusCode == 200) {
        // Log successful OCR results from backend processing
        if (kDebugMode) {
          debugPrint('=== OCR SUCCESS ===');
          debugPrint('Extracted text: ${jsonResponse['text']}');
          debugPrint('Confidence: ${jsonResponse['confidence']}');
          debugPrint('Language: ${jsonResponse['language']}');
          debugPrint('Model used: ${jsonResponse['model_used']}');
          debugPrint('===================');
        }

        // Update UI with OCR results from backend response
        setState(() {
          _extractedText = jsonResponse['text']; // Text extracted by Qwen model
          _confidence = jsonResponse['confidence']?.toDouble(); // Confidence score
          _rawJsonResponse = responseData; // Store raw JSON for debugging
          _isProcessing = false;
        });

        _showSuccess('Text extracted successfully!');
      } else {
        // Handle HTTP error responses from backend (4xx, 5xx status codes)
        String errorMsg = jsonResponse['error'] ?? 'Unknown error occurred';

        // Log error details from backend response
        if (kDebugMode) {
          debugPrint('=== OCR ERROR ===');
          debugPrint('Error message: $errorMsg');
          debugPrint('Status code: ${response.statusCode}');
          debugPrint('Full response: $jsonResponse');
          debugPrint('================');
        }

        setState(() {
          _errorMessage = errorMsg;
          _isProcessing = false;
        });
        _showError(errorMsg);
      }
    } catch (e) {
      // Handle network/connection errors when HTTP request fails
      // This catches issues like: server not running, network timeout, etc.
      if (kDebugMode) {
        debugPrint('=== CONNECTION ERROR ===');
        debugPrint('Error type: ${e.runtimeType}');
        debugPrint('Error message: $e');
        debugPrint('Stack trace: ${StackTrace.current}');
        debugPrint('=======================');
      }
      setState(() {
        _errorMessage = 'Connection error: $e';
        _isProcessing = false;
      });
      _showError('Connection error: $e');
    }
  }

  void _showError(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 4),
        ),
      );
    }
  }

  void _showSuccess(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.green,
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  // Clear all data including HTTP response results and selected file
  // This resets the app state for a new OCR request
  void _clearResults() {
    setState(() {
      _extractedText = null; // Clear OCR result from backend
      _confidence = null; // Clear confidence score from backend
      _errorMessage = null; // Clear any error messages
      _rawJsonResponse = null; // Clear raw JSON response from backend
      _selectedFileBytes = null; // Clear selected image file bytes
      _selectedFileName = null; // Clear selected file name
      _selectedLanguage = null; // Clear selected language
    });
  }

  Widget _buildLanguageButton({
    required String text,
    required String language,
    required VoidCallback onPressed,
    required double width,
    required double height,
  }) {
    bool isSelected = _selectedLanguage == language;

    return SizedBox(
      width: width,
      height: height,
      child: ElevatedButton(
        key: Key(language),
        onPressed: _isProcessing ? null : onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: isSelected ? Colors.teal.shade100 : Colors.white,
          foregroundColor: Colors.black,
          elevation: isSelected ? 4 : 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(40.0),
            side: isSelected
                ? BorderSide(color: Colors.teal, width: 2)
                : BorderSide.none,
          ),
          textStyle: const TextStyle(
            fontFamily: 'Quicksand',
            fontWeight: FontWeight.w500,
          ),
        ),
        child: Text(
          text,
          style: TextStyle(
            fontSize: 16,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
          ),
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    if (_extractedText == null && _errorMessage == null) {
      return const SizedBox.shrink();
    }

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                _errorMessage != null ? 'Error' : 'Extracted Text',
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (_confidence != null)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: _confidence! > 0.7
                        ? Colors.green.shade100
                        : Colors.orange.shade100,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${(_confidence! * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      color: _confidence! > 0.7
                          ? Colors.green.shade800
                          : Colors.orange.shade800,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          if (_errorMessage != null)
            Text(_errorMessage!, style: const TextStyle(color: Colors.red))
          else
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: SelectableText(
                _extractedText!,
                style: const TextStyle(fontSize: 16, height: 1.5),
              ),
            ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TextButton(onPressed: _clearResults, child: const Text('Clear')),
              if (_extractedText != null) ...[
                const SizedBox(width: 8),
                ElevatedButton.icon(
                  onPressed: () {
                    // Copy to clipboard functionality could be added here
                    _showSuccess('Text copied to clipboard!');
                  },
                  icon: const Icon(Icons.copy, size: 16),
                  label: const Text('Copy'),
                ),
                const SizedBox(width: 8),
                // Button to view raw JSON response from OCR API endpoint
                ElevatedButton.icon(
                  onPressed: () {
                    if (_rawJsonResponse != null) {
                      if (kDebugMode) {
                        debugPrint('=== RAW JSON RESPONSE ===');
                        debugPrint(_rawJsonResponse);
                        debugPrint('========================');
                      }
                      _showSuccess('Raw JSON logged to console!');
                    }
                  },
                  icon: const Icon(Icons.code, size: 16),
                  label: const Text('View JSON'),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFB8E6D3),
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Container(
              alignment: Alignment.center,
              margin: const EdgeInsets.only(top: 10.0, left: 8.0, right: 8.0),
              padding: const EdgeInsets.symmetric(
                horizontal: 12.0,
                vertical: 10.0,
              ),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20.0),
              ),
              child: const Text(
                'Gemini-2.0-Flash Handwritten\nImage Reader',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                  height: 1.3,
                ),
              ),
            ),

            // Selected file info
            if (_selectedFileName != null)
              Container(
                margin: const EdgeInsets.all(16),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.image, color: Colors.teal),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _selectedFileName!,
                            style: const TextStyle(fontWeight: FontWeight.w500),
                          ),
                          Text(
                            'Language: $_selectedLanguage',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey.shade600,
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (_isProcessing)
                      const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                  ],
                ),
              ),

            // Language buttons
            if (!_isProcessing)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  children: [
                    const SizedBox(height: 20), // Add margin from title
                    _buildLanguageButton(
                      text: 'English',
                      language: 'english',
                      width: double.infinity,
                      height: 50,
                      onPressed: () => _pickImage('english'),
                    ),
                    const SizedBox(height: 20), // Double the spacing (was 10)
                    _buildLanguageButton(
                      text: 'Français',
                      language: 'french',
                      width: double.infinity,
                      height: 50,
                      onPressed: () => _pickImage('french'),
                    ),
                    const SizedBox(height: 20), // Double the spacing (was 10)
                    _buildLanguageButton(
                      text: 'العربية',
                      language: 'arabic',
                      width: double.infinity,
                      height: 50,
                      onPressed: () => _pickImage('arabic'),
                    ),
                    const SizedBox(height: 20), // Double the spacing (was 10)
                    _buildLanguageButton(
                      text: 'Other Language',
                      language: 'default',
                      width: double.infinity,
                      height: 50,
                      onPressed: () => _pickImage('default'),
                    ),
                  ],
                ),
              ),
            // Processing indicator
            if (_isProcessing)
              Container(
                margin: const EdgeInsets.all(16),
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 16),
                    const Text(
                      'Processing image...',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'This may take a few moments',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey.shade600,
                      ),
                    ),
                  ],
                ),
              ),

            // Results
            Expanded(child: SingleChildScrollView(child: _buildResultCard())),
          ],
        ),
      ),
    );
  }
}