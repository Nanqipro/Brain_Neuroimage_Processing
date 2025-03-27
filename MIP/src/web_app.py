#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
import base64
import io
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from medical_image_processor import MedicalImageProcessor

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'processed')
app.config['TEMPLATES_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMPLATES_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 保存上传的文件
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        base_name, extension = os.path.splitext(filename)
        unique_filename = f"{base_name}_{unique_id}{extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # 返回成功和文件信息
        return jsonify({
            'success': True, 
            'filename': unique_filename,
            'originalName': filename
        })
    
    return jsonify({'error': '不允许的文件类型'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    """处理图像"""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "未找到图像文件"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"})
    
    # 获取处理步骤
    processing_steps = []
    if 'processing_steps' in request.form:
        try:
            processing_steps = json.loads(request.form['processing_steps'])
        except Exception as e:
            return jsonify({"success": False, "error": f"处理步骤解析错误: {str(e)}"})
    
    # 保存上传的文件
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{filename}")
    file.save(temp_file_path)
    
    # 生成唯一ID防止文件覆盖
    unique_id = str(uuid.uuid4())
    output_filename = f"{os.path.splitext(filename)[0]}_{unique_id}_processed{os.path.splitext(filename)[1]}"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        # 读取图像
        img = cv2.imread(temp_file_path)
        if img is None:
            return jsonify({"success": False, "error": "无法读取图像文件"})
        
        # 初始化图像处理器
        processor = MedicalImageProcessor(temp_file_path)
        
        # 处理图像
        if processing_steps and len(processing_steps) > 0:
            # 使用自定义处理步骤
            for step in processing_steps:
                method_name = step.get('method')
                params = step.get('params', {})
                
                # 获取处理方法
                method = getattr(processor, method_name, None)
                if method and callable(method):
                    # 应用处理方法
                    try:
                        method(**params)
                    except Exception as e:
                        print(f"处理步骤 {method_name} 出错: {str(e)}")
        else:
            # 使用默认处理
            processor.apply_gaussian_filter(kernel_size=3)
            processor.apply_bilateral_filter(d=9)
            processor.apply_non_local_means(h=10)
            processor.apply_contrast_enhancement(clip_limit=2.0)
            processor.enhance_bright_spots(kernel_size=5)
            processor.adjust_gamma(gamma=1.2)
            processor.apply_unsharp_masking(amount=1.5)
        
        # 保存处理后的图像
        cv2.imwrite(output_path, processor.processed)
        
        # 转换为base64用于前端显示
        _, original_buffer = cv2.imencode(os.path.splitext(filename)[1], processor.image)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        _, processed_buffer = cv2.imencode(os.path.splitext(filename)[1], processor.processed)
        processed_base64 = base64.b64encode(processed_buffer).decode('utf-8')
        
        # 清理临时文件
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        return jsonify({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "processed_image": f"data:image/png;base64,{processed_base64}"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 清理临时文件
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        return jsonify({"success": False, "error": f"图像处理错误: {str(e)}"})

@app.route('/save_template', methods=['POST'])
def save_template():
    """保存处理模板"""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "未提供模板数据"})
        
        template_name = data.get('template_name')
        steps = data.get('steps')
        
        if not template_name:
            return jsonify({"success": False, "error": "未提供模板名称"})
        
        if not steps:
            return jsonify({"success": False, "error": "未提供处理步骤"})
        
        # 确保文件名安全并添加.json扩展名
        filename = secure_filename(template_name)
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = os.path.join(app.config['TEMPLATES_FOLDER'], filename)
        
        # 写入模板文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(steps, f, ensure_ascii=False, indent=4)
        
        return jsonify({"success": True, "message": f"模板 {template_name} 保存成功"})
    
    except Exception as e:
        return jsonify({"success": False, "error": f"保存模板错误: {str(e)}"})

@app.route('/list_templates', methods=['GET'])
def list_templates():
    """列出所有处理模板"""
    try:
        templates = [f for f in os.listdir(app.config['TEMPLATES_FOLDER']) if f.endswith('.json')]
        return jsonify({"success": True, "templates": templates})
    
    except Exception as e:
        return jsonify({"success": False, "error": f"获取模板列表错误: {str(e)}"})

@app.route('/load_template/<filename>', methods=['GET'])
def load_template(filename):
    """加载处理模板"""
    try:
        file_path = os.path.join(app.config['TEMPLATES_FOLDER'], secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"模板 {filename} 不存在"})
        
        with open(file_path, 'r', encoding='utf-8') as f:
            steps = json.load(f)
        
        return jsonify({"success": True, "steps": steps})
    
    except Exception as e:
        return jsonify({"success": False, "error": f"加载模板错误: {str(e)}"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    """提供处理后的文件"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    """下载处理后的文件"""
    return send_from_directory(
        app.config['PROCESSED_FOLDER'], 
        filename, 
        as_attachment=True
    )

@app.route('/process_cell_image', methods=['POST'])
def process_cell_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "未找到图像文件"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"})
    
    # 保存上传的文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # 生成唯一ID防止文件覆盖
    unique_id = str(uuid.uuid4())
    output_filename = f"{os.path.splitext(filename)[0]}_{unique_id}_processed{os.path.splitext(filename)[1]}"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        # 初始化图像处理器
        processor = MedicalImageProcessor()
        
        # 读取图像
        img = processor.read_image(file_path)
        
        # 处理图像（细胞模式）
        processed_img = processor.process_cell_image(img)
        
        # 保存处理后的图像
        cv2.imwrite(output_path, processed_img)
        
        # 转换为base64用于前端显示
        _, original_buffer = cv2.imencode(os.path.splitext(filename)[1], img)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        _, processed_buffer = cv2.imencode(os.path.splitext(filename)[1], processed_img)
        processed_base64 = base64.b64encode(processed_buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "original_image": f"data:image/png;base64,{original_base64}",
            "processed_image": f"data:image/png;base64,{processed_base64}"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": f"图像处理错误: {str(e)}"})

@app.route('/delete_template', methods=['POST'])
def delete_template():
    try:
        data = request.get_json()
        template_name = data.get('template_name')
        if not template_name:
            return jsonify({'success': False, 'error': '模板名称不能为空'})
        
        template_path = os.path.join(app.config['TEMPLATES_FOLDER'], template_name)
        if not os.path.exists(template_path):
            return jsonify({'success': False, 'error': '模板文件不存在'})
        
        os.remove(template_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def create_app():
    """创建Flask应用"""
    return app

def main():
    """启动Web服务器"""
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 