document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const currentFrame = document.getElementById('current-frame');
    const frameOverlay = document.getElementById('frame-overlay');
    const detectionsList = document.getElementById('detections-list');
    const transcriptionContent = document.getElementById('transcription-content');
    const llmContent = document.getElementById('llm-content');
    const summaryContent = document.getElementById('summary-content');
    const downloadBtn = document.getElementById('download-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const resumeBtn = document.getElementById('resume-btn');
    const fpsStats = document.getElementById('fps-stat');
    const progressStats = document.getElementById('progress-stat');
    
    // Processing state
    let isPaused = false;
    let processingComplete = false;
    let videoComplete = false;
    let transcriptionComplete = false;
    let lastFrameTime = Date.now();
    let frameCount = 0;
    let fpsValue = 0;
    
    // Connect to WebSocket
    const socket = io();
    
    // Set up placeholder
    currentFrame.src = '/static/img/placeholder.html';
    currentFrame.alt = 'Initializing...';
    
    // Start processing when connected
    socket.on('connect', function() {
        console.log('Connected to server');
        
        // Start processing
        socket.emit('start_processing', {
            session_id: sessionId,
            video_path: videoPath
        });
    });
    
    // Handle video info
    socket.on('video_info', function(data) {
        console.log('Received video info:', data);
        
        // Update UI with video info
        summaryContent.innerHTML = `
            <div class="video-summary">
                <p><strong>Resolution:</strong> ${data.width}x${data.height}</p>
                <p><strong>Duration:</strong> ${formatTime(data.duration)}</p>
                <p><strong>FPS:</strong> ${data.fps.toFixed(2)}</p>
                <p><strong>Total Frames:</strong> ${data.total_frames}</p>
            </div>
        `;
    });
    
    // Handle processed frames
    socket.on('processed_frame', function(data) {
        if (isPaused) return;
        
        // Calculate FPS
        const now = Date.now();
        const elapsed = now - lastFrameTime;
        lastFrameTime = now;
        
        frameCount++;
        if (frameCount > 5) { // Average over 5 frames
            fpsValue = 1000 / elapsed;
            frameCount = 0;
            fpsStats.textContent = `FPS: ${fpsValue.toFixed(1)}`;
        }
        
        // Update progress
        progressStats.textContent = `Progress: ${data.progress}%`;
        
        // Update frame image
        if (data.frame) {
            const imageData = hexToBytes(data.frame);
            const blob = new Blob([imageData], { type: 'image/jpeg' });
            const imageUrl = URL.createObjectURL(blob);
            
            currentFrame.src = imageUrl;
            currentFrame.alt = `Frame ${data.frame_idx}`;
            
            // Add frame info overlay if there's a face with expression
            if (data.has_face && data.has_expression) {
                // Add a visual indicator that this frame has significant facial expressions
                const frameContainer = currentFrame.parentElement;
                
                // Create or update the indicator
                let indicator = frameContainer.querySelector('.expression-indicator');
                if (!indicator) {
                    indicator = document.createElement('div');
                    indicator.className = 'expression-indicator';
                    frameContainer.appendChild(indicator);
                }
                
                indicator.innerHTML = `
                    <div class="indicator-content">
                        <span class="indicator-icon">👤</span>
                        <span class="indicator-text">Facial Expression Detected</span>
                    </div>
                `;
                
                // Make it visible
                indicator.style.display = 'block';
                
                // Highlight the current frame
                currentFrame.classList.add('highlight-frame');
                
                // Remove highlight after 2 seconds
                setTimeout(() => {
                    currentFrame.classList.remove('highlight-frame');
                    indicator.style.display = 'none';
                }, 2000);
            }
            
            // Clean up previous URL to avoid memory leaks
            setTimeout(() => URL.createObjectURL(imageUrl), 1000);
        }
        
        // Update detections list
        if (data.detections && data.detections.length > 0) {
            updateDetectionsList(data.detections);
        }
    });
    
    // Handle transcription segments
    socket.on('transcription_segment', function(data) {
        const segment = data.segment;
        
        // Create segment element
        const segmentElement = document.createElement('div');
        segmentElement.className = 'transcription-segment';
        segmentElement.id = `segment-${segment.id}`;
        
        // Format time
        const startTime = formatTime(segment.start);
        const endTime = formatTime(segment.end);
        
        // Add content
        segmentElement.innerHTML = `
            <div class="segment-time">${startTime} - ${endTime}</div>
            <div class="segment-text">${segment.text}</div>
        `;
        
        // Remove loading indicator if present
        const loadingIndicator = transcriptionContent.querySelector('.loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
        
        // Add to transcription content
        transcriptionContent.appendChild(segmentElement);
        
        // Scroll to bottom
        transcriptionContent.scrollTop = transcriptionContent.scrollHeight;
    });
    
    // Handle LLM responses
    socket.on('llm_response', function(data) {
        // Create response element
        const responseElement = document.createElement('div');
        responseElement.className = 'llm-response';
        responseElement.id = `llm-response-${data.segment_id}`;
        
        // Check if this is a response triggered by facial expression
        const isFacialExpressionTriggered = data.frame_idx !== undefined;
        
        // Add special class if triggered by facial expression
        if (isFacialExpressionTriggered) {
            responseElement.classList.add('facial-expression-triggered');
        }
        
        // Format timestamp if available
        let timestampInfo = '';
        if (data.timestamp !== undefined) {
            timestampInfo = ` at ${formatTime(data.timestamp)}`;
        }
        
        // Add content with enhanced information
        responseElement.innerHTML = `
            <div class="llm-response-header">
                ${isFacialExpressionTriggered ? 
                    `<span class="expression-trigger-indicator">👤 Facial Expression Detected</span> ` : 
                    ''}
                Analysis for segment ${data.segment_id}${timestampInfo}:
            </div>
            <div class="llm-response-text">${data.response}</div>
        `;
        
        // Remove loading indicator if present
        const loadingIndicator = llmContent.querySelector('.loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
        
        // Add to LLM content
        llmContent.appendChild(responseElement);
        
        // Scroll to bottom
        llmContent.scrollTop = llmContent.scrollHeight;
    });
    
    // Handle video processing complete
    socket.on('video_processing_complete', function(data) {
        console.log('Video processing complete:', data);
        videoComplete = true;
        checkProcessingComplete();
        
        // Update summary
        updateSummary(data);
    });
    
    // Handle transcription complete
    socket.on('transcription_complete', function(data) {
        console.log('Transcription complete:', data);
        transcriptionComplete = true;
        checkProcessingComplete();
    });
    
    // Handle processing errors
    socket.on('processing_error', function(data) {
        console.error('Processing error:', data);
        
        // Show error message
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = `Error in ${data.component} processing: ${data.error}`;
        
        // Add to summary
        summaryContent.appendChild(errorElement);
    });
    
    // Handle pause/resume buttons
    pauseBtn.addEventListener('click', function() {
        isPaused = true;
        pauseBtn.disabled = true;
        resumeBtn.disabled = false;
    });
    
    resumeBtn.addEventListener('click', function() {
        isPaused = false;
        pauseBtn.disabled = false;
        resumeBtn.disabled = true;
    });
    
    // Handle download button
    downloadBtn.addEventListener('click', function() {
        window.location.href = `/download/${sessionId}`;
    });
    
    // Helper function to update detections list
    function updateDetectionsList(detections) {
        // Clear previous detections
        detectionsList.innerHTML = '';
        
        if (detections.length === 0) {
            detectionsList.innerHTML = '<div class="no-detections">No objects detected</div>';
            return;
        }
        
        // Group detections by class
        const detectionsByClass = {};
        
        detections.forEach(det => {
            const className = det.class_name;
            if (!detectionsByClass[className]) {
                detectionsByClass[className] = [];
            }
            detectionsByClass[className].push(det);
        });
        
        // Create detection items
        for (const className in detectionsByClass) {
            const dets = detectionsByClass[className];
            const count = dets.length;
            
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            
            // Get average confidence
            const avgConfidence = dets.reduce((sum, det) => sum + det.confidence, 0) / count;
            
            detectionItem.innerHTML = `
                <span class="detection-class">${className}</span> (${count})
                <span class="detection-confidence">Confidence: ${(avgConfidence * 100).toFixed(1)}%</span>
            `;
            
            detectionsList.appendChild(detectionItem);
        }
    }
    
    // Helper function to check if processing is complete
    function checkProcessingComplete() {
        if (videoComplete && transcriptionComplete) {
            processingComplete = true;
            downloadBtn.disabled = false;
            
            // Show completion message
            const completionElement = document.createElement('div');
            completionElement.className = 'success-message';
            completionElement.textContent = 'Processing complete! You can now download the summary.';
            
            summaryContent.appendChild(completionElement);
        }
    }
    
    // Helper function to update summary
    function updateSummary(data) {
        const summaryElement = document.createElement('div');
        summaryElement.className = 'processing-summary';
        
        summaryElement.innerHTML = `
            <p><strong>Processed Frames:</strong> ${data.processed_frames}</p>
            <p><strong>Total Detections:</strong> ${data.total_detections}</p>
        `;
        
        // Remove loading indicator if present
        const loadingIndicator = summaryContent.querySelector('.loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
        
        // Add to summary content
        summaryContent.appendChild(summaryElement);
    }
    
    // Helper function to format time
    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        
        return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
    }
    
    // Helper function to convert hex string to bytes
    function hexToBytes(hex) {
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < hex.length; i += 2) {
            bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
        }
        return bytes;
    }
});
