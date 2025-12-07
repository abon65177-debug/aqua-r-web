import cv2
import os
import sys
import numpy as np
import random
import math
from datetime import datetime
from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ==========================================
# 1. إعدادات السيرفر
# ==========================================
app = Flask(__name__)
app.secret_key = 'aqua_r_cloud_key'

# مجلدات الرفع والنتائج
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# تحميل الموديلات (مع حماية ضد الأخطاء)
MODEL_WATER_PATH = "water_hyacinth.pt"
MODEL_RUBBISH_PATH = "rubbish.pt"
model_water = None
model_rubbish = None

print("🔄 Loading AI Models...")
if os.path.exists(MODEL_WATER_PATH) and os.path.exists(MODEL_RUBBISH_PATH):
    try:
        model_water = YOLO(MODEL_WATER_PATH)
        model_rubbish = YOLO(MODEL_RUBBISH_PATH)
        print("✅ Models Loaded!")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
else:
    print("⚠️ Models not found. Please upload .pt files.")

# بيانات الروبوت الوهمية
robot_status = {
    "battery": 92,
    "status": "Ready",
    "lat": 30.0444, "lng": 31.2357,
    "trash_count": 12
}

# --- قاموس الترجمة (مختصر) ---
TRANSLATIONS = {
    'en': {'dir':'ltr', 'title':'AQUA-R', 'dashboard':'Dashboard', 'upload':'Upload & Analyze', 'result':'Analysis Result'},
    'ar': {'dir':'rtl', 'title':'أكوا-آر', 'dashboard':'لوحة التحكم', 'upload':'رفع وتحليل', 'result':'نتيجة التحليل'},
    'fr': {'dir':'ltr', 'title':'AQUA-R', 'dashboard':'Tableau de bord', 'upload':'Télécharger', 'result':'Résultat'}
}

# ==========================================
# 2. دوال المعالجة (AI Processing)
# ==========================================
def process_image(filepath):
    if not model_water or not model_rubbish: return filepath # لو مفيش موديل رجع الصورة زي ما هي
    
    img = cv2.imread(filepath)
    if img is None: return filepath

    # تشغيل الموديلات
    res_w = model_water(img, conf=0.25)[0]
    res_r = model_rubbish(img, conf=0.25)[0]
    
    # رسم النتائج
    # 1. ورد النيل (أخضر)
    for box in res_w.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(img, "Water Hyacinth", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
    # 2. القمامة (أحمر)
    for box in res_r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.putText(img, "Trash", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        robot_status['trash_count'] += 1

    # حفظ النتيجة
    filename = os.path.basename(filepath)
    result_path = os.path.join(app.config['RESULTS_FOLDER'], 'pred_' + filename)
    cv2.imwrite(result_path, img)
    return 'pred_' + filename

# ==========================================
# 3. واجهة المستخدم (Dashboard HTML)
# ==========================================
HTML_DASHBOARD = """
<!DOCTYPE html>
<html dir="{{ t['dir'] }}">
<head>
    <title>AQUA-R Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Cairo&family=Orbitron&display=swap" rel="stylesheet">
    <style>
        body { background: #050a14; color: white; font-family: 'Cairo', sans-serif; margin:0; padding:20px; text-align:center; }
        .card { background: rgba(20,30,50,0.9); padding: 20px; border-radius: 15px; border: 1px solid #00f2ff; margin-bottom: 20px; max-width: 600px; margin-left: auto; margin-right: auto; }
        h1 { color: #00f2ff; font-family: 'Orbitron'; }
        img { max-width: 100%; border-radius: 10px; border: 2px solid #333; }
        .btn { background: #00f2ff; color: #000; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; width: 100%; margin-top:10px; }
        input[type=file] { margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>🤖 AQUA-R AI VISION</h1>
    
    <!-- Upload Section -->
    <div class="card">
        <h3>📤 {{ t['upload'] }}</h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit" class="btn">🚀 Analyze Now</button>
        </form>
    </div>

    <!-- Result Section -->
    <div class="card">
        <h3>👁️ {{ t['result'] }}</h3>
        {% if image_file %}
            <img src="{{ url_for('static', filename='results/' + image_file) }}">
            <p style="color:#0f0;">Analysis Complete ✅</p>
        {% else %}
            <div style="height:200px; background:#111; display:flex; align-items:center; justify-content:center; color:#555; border-radius:10px;">
                Waiting for image...
            </div>
        {% endif %}
    </div>

    <!-- Stats -->
    <div class="card">
        <h3>📊 Stats</h3>
        <p>🔋 Battery: {{ status.battery }}% | 🗑️ Trash: {{ status.trash_count }}</p>
    </div>
</body>
</html>
"""

# ==========================================
# 4. المسارات (Routes)
# ==========================================
@app.route('/')
def index():
    # اللغة الافتراضية
    lang = session.get('lang', 'en')
    t = TRANSLATIONS[lang]
    # آخر صورة تمت معالجتها
    img = session.get('last_result', None)
    return render_template_string(HTML_DASHBOARD, t=t, image_file=img, status=robot_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return redirect('/')
    file = request.files['file']
    if file.filename == '': return redirect('/')
    
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # معالجة الصورة بالذكاء الاصطناعي
        result_filename = process_image(save_path)
        session['last_result'] = result_filename
        
        return redirect('/')

# لتشغيل السيرفر
if __name__ == '__main__':
    # استخدم المنفذ الذي يحدده Render أو 5000 كاحتياطي
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)