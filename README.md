# Smart Attendance System with WhatsApp Notifications

A facial recognition-based attendance tracking system that automatically detects students using camera feed and sends WhatsApp notifications to parents when their child arrives.

## Features

### 🎯 Core Features
- **Real-time Facial Recognition**: Uses advanced FaceNet model for accurate face detection and recognition
- **Automatic Attendance Tracking**: Records attendance with timestamps when students are detected
- **WhatsApp Notifications**: Automatically sends notifications to parents via WhatsApp when their child arrives
- **Multi-day Attendance Management**: Supports tracking attendance across multiple days with date-specific columns
- **Excel Integration**: Stores attendance data in Excel format with easy export capabilities

### 🔧 Management Features
- **Camera Control**: Start/stop camera feed with real-time video display
- **Attendance Display**: View current attendance status in the application
- **Date Management**: Switch between attendance days, reset attendance, and delete days
- **Data Export**: Download attendance data for selected date ranges
- **Student Database**: CSV-based student information with parent contact details

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Webcam/camera device
- WhatsApp installed on the system
- Windows/Linux/Mac OS

### Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data files**:
   - Ensure `dataset.csv` contains student information with columns: `Student_ID`, `Name`, `Image_Dir`, `S_gender`, `S_parents_number`, `S_Dept`, `S_semester`
   - Place student images in the `images/` directory with subdirectories for each student (e.g., `images/001_student_name/`)

4. **Generate face encodings**:
   ```bash
   python encode_faces.py
   ```
   This will create `encodings.pkl` file containing face embeddings for all students.

### Usage

1. **Run the main application**:
   ```bash
   python main.py
   ```

2. **Initial Setup**:
   - The application will automatically create `attendance.xlsx` if it doesn't exist
   - Face encodings will be loaded from `encodings.pkl`

3. **Taking Attendance**:
   - Click "📸 Start Camera" to begin facial recognition
   - Students will be automatically detected and marked as present with timestamps
   - Parents will receive WhatsApp notifications when their child arrives
   - Click "🛑 Stop Camera" to stop the recognition process

4. **Managing Attendance**:
   - Use "🗓️ Next Day" to switch to the next attendance day
   - "🧹 Reset Current Day" to clear attendance for the current day
   - "🗑️ Delete Current Day" to remove the current day's column
   - "📊 Show Attendance List" to view current attendance data
   - "⬇ Download Excel" to export attendance data for selected dates

### Data Files Structure

- `dataset.csv`: Student information with contact details
- `attendance.xlsx`: Attendance records organized by dates
- `encodings.pkl`: Pre-computed face embeddings for recognition
- `images/`: Directory containing student photos organized by student ID

### WhatsApp Notification Setup

The system automatically sends WhatsApp messages to parents using:
- Phone numbers from `dataset.csv` (S_parents_number column)
- Automatic WhatsApp URL launching and message pasting
- Notifications include student name and arrival time

### Recognition Settings

- **Recognition Threshold**: Set to 0.9 (configurable in code)
- **Face Detection**: Uses MTCNN for robust face detection
- **Embedding Model**: FaceNet (InceptionResnetV1) pretrained on VGGFace2

## Troubleshooting

- Ensure camera permissions are granted
- Check that `encodings.pkl` exists and contains valid data
- Verify WhatsApp is installed and phone numbers are in correct format (+92xxxxxxxxxx)
- Make sure all required Python packages are installed

## Requirements

See `requirements.txt` for complete list of Python dependencies including:
- OpenCV for computer vision
- PyTorch for deep learning
- Face recognition libraries
- GUI and automation tools
