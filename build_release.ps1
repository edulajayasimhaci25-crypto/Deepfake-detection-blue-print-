Write-Host "Packaging Production Deepfake Detection API Engine..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "release_bundle" | Out-Null
Copy-Item "server.py" -Destination "release_bundle/"
Copy-Item "vision_model.onnx" -Destination "release_bundle/"
Copy-Item "audio_model.onnx" -Destination "release_bundle/"
Copy-Item "requirements.txt" -Destination "release_bundle/"
Copy-Item "Dockerfile" -Destination "release_bundle/"
Copy-Item "benchmark.py" -Destination "release_bundle/"
Compress-Archive -Path "release_bundle\*" -DestinationPath "deepfake_detector_v1.0.zip" -Force
Write-Host "Success: Compressed deployment package 'deepfake_detector_v1.0.zip' generated." -ForegroundColor Cyan
