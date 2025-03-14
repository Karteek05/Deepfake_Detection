## **AI-Based Deepfake Detection System 🔍🎯**

### **Overview**
This project is an **AI-powered deepfake detection system** designed to identify manipulated images and videos using advanced **Deep Learning** techniques. By leveraging powerful neural networks and visual explanation methods like **Grad-CAM**, this solution offers both detection accuracy and explainability.

### **Key Features 🚀**
✅ Detects deepfake content with improved accuracy using a custom-trained model.  
✅ Uses **Grad-CAM heatmaps** to visualize manipulated areas in deepfake images/videos.  
✅ Supports both **image** and **video** uploads for comprehensive detection.  
✅ Features a user-friendly web interface built with **Gradio** for easy interaction.  
✅ Includes **temperature scaling** to improve prediction confidence scores.  

---

### **Tech Stack 🛠️**
- **Python** (Core Language)  
- **TensorFlow/Keras** (Deep Learning Framework)  
- **OpenCV** (Image/Video Processing)  
- **Gradio** (Web Interface for Real-Time Detection)  
- **PIL** (Image Handling)  

---

### **Model Architecture 🧠**
- Base Model: **Xception Network** (Pre-trained on ImageNet)  
- Fine-tuned with **FaceForensics++** and **DFDC** datasets  
- Last convolutional layers unfreezed for better feature learning  
- Optimized with **temperature scaling** to reduce overconfident false positives  

---

### **How to Run the Project ▶️**

1. **Install Dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Download the Model File**  
   Add your trained model file (`deepfake_detector_fixed.h5`) to the project directory.

3. **Run the Gradio Web Interface**  
   ```
   python web.py
   ```

4. **Access the Interface**  
   - Gradio will provide a **local link** (e.g., `http://localhost:7860`) and a **public share link** for testing.

---

### **Usage Instructions 📋**
- Upload an **image** or **video** via the web interface.  
- The system will display:  
  - **Prediction** → "Real" or "Fake"  
  - **Confidence Scores** → Probability of the content being real or fake  
  - **Grad-CAM Heatmap** → Highlights manipulated regions in fake content  

---

### **Limitations ⚠️**
- The system may struggle with:  
  - **Extremely low-quality content**  
  - **Sophisticated deepfakes** with minimal visual artifacts  
- Confidence scores may vary for borderline cases (e.g., 49% Fake / 51% Real).

---

### **Future Improvements 🔄**
✅ Enhanced dataset integration using **DFDC** for improved robustness.  
✅ Improved UI with a more dynamic, parameter-driven experience.  
✅ Potential integration with **Flask/Django** for scalable deployment.  

---

### **Contributors 👥**
- Karteek Cherukupalli
- Open for collaborations — feel free to fork, star ⭐, or submit issues!

---

### **License 📄**
This project is licensed under the MIT License — feel free to use, modify, and contribute.
