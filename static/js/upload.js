document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('video-upload');
    const fileName = document.getElementById('file-name');
    const uploadProgress = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    // Update file name when file is selected
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileName.textContent = this.files[0].name;
            
            // Show file size
            const fileSize = this.files[0].size;
            const fileSizeFormatted = formatFileSize(fileSize);
            fileName.textContent += ` (${fileSizeFormatted})`;
        } else {
            fileName.textContent = 'No file selected';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            showError('Please select a video file to upload');
            return;
        }
        
        const file = fileInput.files[0];
        
        // Check if file is a video
        if (!file.type.startsWith('video/')) {
            showError('Please select a valid video file');
            return;
        }
        
        // Create FormData object
        const formData = new FormData();
        formData.append('video', file);
        
        // Show progress bar
        uploadProgress.style.display = 'block';
        
        // Create and configure XMLHttpRequest
        const xhr = new XMLHttpRequest();
        
        // Track upload progress
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                progressFill.style.width = percentComplete + '%';
                progressText.textContent = `Uploading: ${percentComplete}%`;
            }
        });
        
        // Handle response
        xhr.addEventListener('load', function() {
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    
                    if (response.success) {
                        progressText.textContent = 'Upload complete! Redirecting...';
                        
                        // Redirect to processing page
                        setTimeout(function() {
                            window.location.href = response.redirect;
                        }, 1000);
                    } else {
                        showError(response.error || 'Upload failed');
                    }
                } catch (e) {
                    showError('Invalid response from server');
                }
            } else {
                showError(`Upload failed with status ${xhr.status}`);
            }
        });
        
        // Handle errors
        xhr.addEventListener('error', function() {
            showError('Upload failed due to network error');
        });
        
        xhr.addEventListener('abort', function() {
            showError('Upload aborted');
        });
        
        // Send the request
        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    });
    
    // Helper function to format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Helper function to show error message
    function showError(message) {
        // Remove any existing error messages
        const existingErrors = document.querySelectorAll('.error-message');
        existingErrors.forEach(el => el.remove());
        
        // Create error message element
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        
        // Insert after form
        uploadForm.parentNode.insertBefore(errorElement, uploadForm.nextSibling);
        
        // Hide progress bar
        uploadProgress.style.display = 'none';
        
        // Auto-remove after 5 seconds
        setTimeout(function() {
            errorElement.remove();
        }, 5000);
    }
});
