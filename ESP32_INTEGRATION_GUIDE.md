# ESP32 Educational Poster Integration Guide

This guide explains how to connect your HTML frontend to the FastAPI backend for sending educational posters to ESP32 devices.

## 🔗 What's Been Implemented

### Backend Changes (`app/main.py`)
- ✅ **New ESP32Request model** - Accepts specific image IDs
- ✅ **Enhanced /send_to_esp32 endpoint** - Now accepts the exact image to send
- ✅ **Better error handling** - Validates image exists before sending
- ✅ **Test endpoint** - `/api/test-esp32` for connection testing

### Frontend Changes (`app/static/js/script.js`)
- ✅ **Enhanced sendToESP32 function** - Sends specific image ID to backend
- ✅ **Visual feedback** - Loading, success, and error states for the button
- ✅ **Better error messages** - Detailed feedback for troubleshooting
- ✅ **Test button** - New "Test ESP32" button to verify API connectivity

### CSS Changes (`app/static/css/styles.css`)
- ✅ **Button animations** - Smooth transitions and feedback states
- ✅ **Success/error styling** - Color-coded feedback for user actions

## 🚀 How to Test the Integration

### 1. Start Your Backend Server
```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test the Web Interface
1. Open `http://localhost:8000` in your browser
2. Generate an educational poster:
   - Enter a topic (e.g., "Photosynthesis")
   - Select age group
   - Click "Generate Educational Poster"
3. Once generated, you'll see the **"Send to ESP32"** button
4. Click **"Test ESP32"** first to verify API connectivity
5. Click **"Send to ESP32"** to queue the image for your ESP32

### 3. API Endpoints for ESP32

Your ESP32 needs to call these endpoints:

#### Check for Updates
```http
GET /check_update
Response: {"update": true/false}
```

#### Fetch Latest Poster
```http
GET /latest
Response: {
  "filename": "poster_name",
  "width": 128,
  "height": 64,
  "data": ["0x00", "0x01", ...],
  "size": 1024,
  "timestamp": 1234567890
}
```

#### Test API (For Debugging)
```http
GET /api/test-esp32
Response: {
  "status": "API working",
  "available_posters": 2,
  "poster_ids": ["topic1_7-12_flux_123", "topic2_0-6_flux_456"],
  "pending_image": "topic1_7-12_flux_123"
}
```

## 🔧 ESP32 Setup

### Hardware Required
- ESP32 development board
- 128x64 OLED display (SSD1306)
- Breadboard and jumper wires

### Wiring
```
ESP32    OLED Display
GPIO21 → SDA
GPIO22 → SCL  
3.3V   → VCC
GND    → GND
```

### Arduino Libraries Required
Install these through Arduino IDE Library Manager:
- **ArduinoJson** by Benoit Blanchon
- **U8g2** by oliver
- **WiFi** (included with ESP32 core)
- **HTTPClient** (included with ESP32 core)

### Configuration Steps
1. Open `esp32_example.ino` in Arduino IDE
2. Update these lines with your information:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverURL = "http://YOUR_COMPUTER_IP:8000";
   ```
3. Find your computer's IP address:
   - Windows: `ipconfig` in Command Prompt
   - Mac/Linux: `ifconfig` in Terminal
4. Upload the code to your ESP32

## 🧪 Testing the Complete Workflow

### Step 1: Backend Test
```bash
# Test the API directly
curl http://localhost:8000/api/test-esp32
```

### Step 2: Generate a Poster
1. Open web interface
2. Generate an educational poster
3. Verify it appears in the preview

### Step 3: Send to ESP32
1. Click "Send to ESP32" button
2. Should see success message
3. Check backend logs for confirmation

### Step 4: ESP32 Fetch
Your ESP32 will automatically:
1. Check for updates every 5 seconds
2. Fetch new poster when available
3. Display it on the OLED screen

## 🐛 Troubleshooting

### Frontend Issues
- **Button doesn't respond**: Check browser console for JavaScript errors
- **"Image not found" error**: Ensure you generated a poster first
- **API test fails**: Verify backend is running on port 8000

### Backend Issues
- **Server won't start**: Check for port conflicts or missing dependencies
- **CORS errors**: Add CORS middleware if accessing from different domain

### ESP32 Issues
- **Won't connect to WiFi**: Verify SSID/password are correct
- **Can't reach server**: Check IP address and ensure both devices are on same network
- **Display not working**: Verify OLED wiring and I2C address
- **JSON parsing errors**: Increase buffer size in ESP32 code if needed

### Network Issues
- **Connection timeout**: Ensure firewall allows port 8000
- **Wrong IP address**: Use `ipconfig`/`ifconfig` to find correct IP
- **Router blocking**: Some routers block device-to-device communication

## 📝 How It Works

1. **User generates poster** on web interface
2. **Backend creates bitmap** optimized for 128x64 OLED
3. **User clicks "Send to ESP32"** → frontend sends image ID to `/send_to_esp32`
4. **Backend marks image as pending** in global `pending_image` variable
5. **ESP32 polls `/check_update`** every 5 seconds
6. **When update available**, ESP32 fetches from `/latest` endpoint
7. **ESP32 displays educational poster** on OLED screen

## 🎯 Next Steps

- Add multiple ESP32 support with device IDs
- Implement image queuing system
- Add poster scheduling functionality
- Create mobile app interface
- Add poster library/history feature

## 📚 Educational Use Cases

Perfect for:
- **Classroom displays** - Instant educational content updates
- **Science fairs** - Dynamic poster presentations  
- **Museum exhibits** - Rotating educational information
- **Homeschooling** - Interactive learning displays
- **Libraries** - Featured topic displays

The system is now fully integrated and ready for educational use! 🎓