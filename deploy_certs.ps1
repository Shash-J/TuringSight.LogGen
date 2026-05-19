Write-Host "=== TuringSight Certificate Deployer ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Step 1: Creating certs directory on server..." -ForegroundColor Yellow
Write-Host "  You will be asked for passwords TWICE: first 'rvce', then 'rvce2024'"
Write-Host ""

# Create directory on remote
ssh -o StrictHostKeyChecking=no -J rvce@172.16.18.6 rvce@172.16.18.3 "mkdir -p ~/fresh_turingsight/TuringSight.LogGen/'RunPod log gen'/configs/certs"

Write-Host ""
Write-Host "Step 2: Copying AmazonRootCA1.pem..." -ForegroundColor Yellow
scp -o StrictHostKeyChecking=no -o ProxyJump=rvce@172.16.18.6 "RunPod log gen\configs\certs\AmazonRootCA1.pem" "rvce@172.16.18.3:fresh_turingsight/TuringSight.LogGen/RunPod log gen/configs/certs/"

Write-Host "Step 3: Copying certificate.pem.crt..." -ForegroundColor Yellow
scp -o StrictHostKeyChecking=no -o ProxyJump=rvce@172.16.18.6 "RunPod log gen\configs\certs\certificate.pem.crt" "rvce@172.16.18.3:fresh_turingsight/TuringSight.LogGen/RunPod log gen/configs/certs/"

Write-Host "Step 4: Copying private.pem.key..." -ForegroundColor Yellow
scp -o StrictHostKeyChecking=no -o ProxyJump=rvce@172.16.18.6 "RunPod log gen\configs\certs\private.pem.key" "rvce@172.16.18.3:fresh_turingsight/TuringSight.LogGen/RunPod log gen/configs/certs/"

Write-Host ""
Write-Host "Step 5: Verifying and pulling latest code..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no -J rvce@172.16.18.6 rvce@172.16.18.3 "ls -la ~/fresh_turingsight/TuringSight.LogGen/'RunPod log gen'/configs/certs/ && cd ~/fresh_turingsight/TuringSight.LogGen/'RunPod log gen' && git pull origin main"

Write-Host ""
Write-Host "=== DONE! Certs deployed and code updated. ===" -ForegroundColor Green
Write-Host "Now go to your SSH terminal and run:" -ForegroundColor Cyan
Write-Host '  cd ~/fresh_turingsight/TuringSight.LogGen/"RunPod log gen"'
Write-Host '  export RTSP_URL="rtsp://admin:Tech@007@106.51.57.8:554/cam/realmonitor?channel=1&subtype=0"'
Write-Host '  python main.py'
Write-Host ""
Read-Host "Press Enter to close"
