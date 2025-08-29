package com.example.flutter_qwen_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.content.Intent
import android.app.Activity
import android.net.Uri
import android.provider.DocumentsContract
import androidx.documentfile.provider.DocumentFile

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.flutter_qwen_app/file_picker"
    private val REQUEST_CODE_PICK_FOLDER = 1001

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "pickFolder" -> {
                    pickFolder(result)
                }
                "pickImages" -> {
                    pickImages(result)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    private fun pickFolder(result: MethodChannel.Result) {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        }
        
        try {
            startActivityForResult(intent, REQUEST_CODE_PICK_FOLDER)
            // Store the result callback for later use
            pendingResult = result
        } catch (e: Exception) {
            result.error("FOLDER_PICKER_ERROR", "Failed to open folder picker: ${e.message}", null)
        }
    }

    private fun pickImages(result: MethodChannel.Result) {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        
        try {
            val chooser = Intent.createChooser(intent, "Select Images")
            startActivityForResult(chooser, REQUEST_CODE_PICK_IMAGES)
            pendingResult = result
        } catch (e: Exception) {
            result.error("IMAGE_PICKER_ERROR", "Failed to open image picker: ${e.message}", null)
        }
    }

    private var pendingResult: MethodChannel.Result? = null
    companion object {
        private const val REQUEST_CODE_PICK_IMAGES = 1002
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        when (requestCode) {
            REQUEST_CODE_PICK_FOLDER -> {
                if (resultCode == Activity.RESULT_OK && data != null) {
                    val uri = data.data
                    if (uri != null) {
                        // Take persistable permission
                        contentResolver.takePersistableUriPermission(
                            uri,
                            Intent.FLAG_GRANT_READ_URI_PERMISSION
                        )
                        
                        val documentFile = DocumentFile.fromTreeUri(this, uri)
                        val imagePaths = mutableListOf<String>()
                        
                        documentFile?.listFiles()?.forEach { file ->
                            if (file.isFile && isImageFile(file.name)) {
                                imagePaths.add(file.uri.toString())
                            }
                        }
                        
                        pendingResult?.success(imagePaths)
                    } else {
                        pendingResult?.error("NO_URI", "No folder URI received", null)
                    }
                } else {
                    pendingResult?.error("CANCELLED", "Folder selection cancelled", null)
                }
            }
            
            REQUEST_CODE_PICK_IMAGES -> {
                if (resultCode == Activity.RESULT_OK && data != null) {
                    val imagePaths = mutableListOf<String>()
                    
                    if (data.clipData != null) {
                        // Multiple images selected
                        val clipData = data.clipData!!
                        for (i in 0 until clipData.itemCount) {
                            val uri = clipData.getItemAt(i).uri
                            imagePaths.add(uri.toString())
                        }
                    } else if (data.data != null) {
                        // Single image selected
                        imagePaths.add(data.data.toString())
                    }
                    
                    pendingResult?.success(imagePaths)
                } else {
                    pendingResult?.error("CANCELLED", "Image selection cancelled", null)
                }
            }
        }
        
        pendingResult = null
    }

    private fun isImageFile(fileName: String?): Boolean {
        if (fileName == null) return false
        val imageExtensions = listOf(".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
        return imageExtensions.any { fileName.lowercase().endsWith(it) }
    }
}