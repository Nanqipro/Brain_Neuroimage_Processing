/**
 * 医学图像处理Web应用主脚本
 */
document.addEventListener('DOMContentLoaded', function() {
    // 初始化步骤管理器
    const stepsManager = new ProcessingStepsManager();
    
    // 状态管理
    let currentImageFile = null;
    let isProcessing = false;
    let processingTimeout = null;
    const PROCESSING_DELAY = 500; // 防抖延迟时间，单位毫秒
    
    // DOM元素缓存
    const elements = {
        uploadArea: document.getElementById('upload-area'),
        imageFile: document.getElementById('image-file'),
        originalImage: document.getElementById('original-image'),
        processedImage: document.getElementById('processed-image'),
        resultsSection: document.getElementById('results-section'),
        processingStepsList: document.getElementById('processing-steps-list'),
        addStepBtn: document.getElementById('add-step-btn'),
        availableMethodsSelect: document.getElementById('available-methods'),
        templateNameInput: document.getElementById('template-name'),
        saveTemplateBtn: document.getElementById('save-template-btn'),
        templatesSelect: document.getElementById('templates-select'),
        loadTemplateBtn: document.getElementById('load-template-btn'),
        downloadBtn: document.getElementById('download-btn'),
        tooltipTemplate: document.getElementById('tooltip-template')
    };

    // 隐藏结果区域，直到有图像上传
    elements.resultsSection.style.display = 'none';
    
    // 创建工具提示元素
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    document.body.appendChild(tooltip);
    
    // 显示工具提示
    function showTooltip(target, title, description, params) {
        const rect = target.getBoundingClientRect();
        const tooltipContent = elements.tooltipTemplate.querySelector('.tooltip-content').cloneNode(true);
        
        tooltipContent.querySelector('.tooltip-title').textContent = title;
        tooltipContent.querySelector('.tooltip-description').textContent = description;
        
        // 添加参数说明
        const paramsContainer = tooltipContent.querySelector('.tooltip-params');
        if (params && Object.keys(params).length > 0) {
            paramsContainer.innerHTML = '<div class="param-heading">参数说明：</div>';
            
            for (const key in params) {
                const paramInfo = document.createElement('div');
                paramInfo.innerHTML = `<b>${params[key].name}</b>: ${params[key].description}`;
                paramsContainer.appendChild(paramInfo);
            }
        } else {
            paramsContainer.style.display = 'none';
        }
        
        tooltip.innerHTML = '';
        tooltip.appendChild(tooltipContent);
        
        // 定位工具提示
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.top = `${rect.bottom + 5}px`;
        
        // 显示工具提示
        tooltip.classList.add('visible');
    }
    
    // 隐藏工具提示
    function hideTooltip() {
        tooltip.classList.remove('visible');
    }
    
    // 更新滑块渐变
    function updateSliderBackground(slider) {
        const min = parseFloat(slider.min);
        const max = parseFloat(slider.max);
        const value = parseFloat(slider.value);
        const percentage = ((value - min) / (max - min)) * 100;
        
        slider.style.background = `linear-gradient(to right, #3498db 0%, #3498db ${percentage}%, #ddd ${percentage}%, #ddd 100%)`;
    }

    // 初始化时填充可用方法下拉列表
    function initializeAvailableMethods() {
        const methods = stepsManager.getSupportedMethods();
        elements.availableMethodsSelect.innerHTML = '';
        
        methods.forEach(method => {
            const option = document.createElement('option');
            option.value = method.id;
            option.textContent = method.name;
            elements.availableMethodsSelect.appendChild(option);
        });
    }

    // 渲染处理步骤列表
    function renderProcessingSteps() {
        const steps = stepsManager.getSteps();
        elements.processingStepsList.innerHTML = '';
        
        if (steps.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.classList.add('empty-message');
            emptyMessage.textContent = '尚未添加处理步骤。请点击"添加步骤"按钮添加处理步骤。';
            elements.processingStepsList.appendChild(emptyMessage);
            return;
        }
        
        steps.forEach((step, index) => {
            const methodInfo = stepsManager.getMethodInfo(step.method);
            if (!methodInfo) return;
            
            const stepElement = document.createElement('div');
            stepElement.classList.add('processing-step');
            stepElement.dataset.index = index;
            
            // 创建拖动手柄
            const dragHandle = document.createElement('div');
            dragHandle.classList.add('drag-handle');
            dragHandle.innerHTML = '⋮⋮';
            stepElement.appendChild(dragHandle);
            
            // 创建步骤标题和删除按钮
            const headerDiv = document.createElement('div');
            headerDiv.classList.add('step-header');
            
            const stepTitle = document.createElement('h4');
            const toggleIcon = document.createElement('span');
            toggleIcon.classList.add('step-toggle');
            toggleIcon.textContent = '▼';
            stepTitle.appendChild(toggleIcon);
            stepTitle.appendChild(document.createTextNode(`${index + 1}. ${methodInfo.name}`));
            headerDiv.appendChild(stepTitle);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.classList.add('delete-step-btn');
            deleteBtn.textContent = '删除';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation(); // 防止触发折叠事件
                if (confirm(`确定要删除"${methodInfo.name}"处理步骤吗？`)) {
                    stepsManager.removeStep(index);
                    renderProcessingSteps();
                    
                    // 如果有图像，则重新处理
                    if (currentImageFile) {
                        scheduleProcessing();
                    }
                }
            });
            headerDiv.appendChild(deleteBtn);
            
            stepElement.appendChild(headerDiv);
            
            // 创建步骤内容容器
            const contentDiv = document.createElement('div');
            contentDiv.classList.add('step-content');
            
            // 添加步骤描述
            const stepDesc = document.createElement('p');
            stepDesc.classList.add('step-description');
            stepDesc.textContent = methodInfo.description;
            contentDiv.appendChild(stepDesc);
            
            // 创建参数设置区域
            const paramsDiv = document.createElement('div');
            paramsDiv.classList.add('step-params');
            
            for (const paramKey in methodInfo.params) {
                const paramInfo = methodInfo.params[paramKey];
                const paramValue = step.params[paramKey] || paramInfo.default;
                
                const paramDiv = document.createElement('div');
                paramDiv.classList.add('param-item');
                
                // 创建参数名称和帮助图标
                const paramName = document.createElement('div');
                paramName.classList.add('param-name');
                
                const paramLabel = document.createElement('label');
                paramLabel.textContent = `${paramInfo.name}`;
                paramName.appendChild(paramLabel);
                
                const helpIcon = document.createElement('span');
                helpIcon.classList.add('help-icon');
                helpIcon.textContent = '?';
                helpIcon.title = paramInfo.description;
                
                // 添加工具提示事件
                helpIcon.addEventListener('mouseenter', () => {
                    showTooltip(helpIcon, paramInfo.name, paramInfo.description);
                });
                
                helpIcon.addEventListener('mouseleave', () => {
                    hideTooltip();
                });
                
                paramName.appendChild(helpIcon);
                paramDiv.appendChild(paramName);
                
                if (paramInfo.type === 'select') {
                    // 下拉选择参数
                    const select = document.createElement('select');
                    
                    paramInfo.options.forEach(option => {
                        const optElement = document.createElement('option');
                        optElement.value = option.value;
                        optElement.textContent = option.text;
                        
                        if (paramValue.toString() === option.value.toString()) {
                            optElement.selected = true;
                        }
                        
                        select.appendChild(optElement);
                    });
                    
                    select.addEventListener('change', (e) => {
                        const params = {};
                        params[paramKey] = e.target.value;
                        stepsManager.updateStepParams(index, params);
                        
                        // 如果有图像，则重新处理
                        if (currentImageFile) {
                            scheduleProcessing();
                        }
                    });
                    
                    paramDiv.appendChild(select);
                } else {
                    // 创建滑块和数字输入框的容器
                    const sliderContainer = document.createElement('div');
                    sliderContainer.classList.add('param-slider-container');
                    
                    // 创建滑块
                    const slider = document.createElement('input');
                    slider.type = 'range';
                    slider.min = paramInfo.min;
                    slider.max = paramInfo.max;
                    slider.step = paramInfo.step;
                    slider.value = paramValue;
                    
                    // 创建数字输入框
                    const inputNumber = document.createElement('input');
                    inputNumber.type = 'number';
                    inputNumber.min = paramInfo.min;
                    inputNumber.max = paramInfo.max;
                    inputNumber.step = paramInfo.step;
                    inputNumber.value = paramValue;
                    
                    // 更新滑块背景
                    updateSliderBackground(slider);
                    
                    // 双向绑定
                    slider.addEventListener('input', (e) => {
                        inputNumber.value = e.target.value;
                        updateSliderBackground(slider);
                        
                        const params = {};
                        params[paramKey] = parseFloat(e.target.value);
                        stepsManager.updateStepParams(index, params);
                        
                        // 如果有图像，则重新处理
                        if (currentImageFile) {
                            scheduleProcessing();
                        }
                    });
                    
                    inputNumber.addEventListener('change', (e) => {
                        slider.value = e.target.value;
                        updateSliderBackground(slider);
                        
                        const params = {};
                        params[paramKey] = parseFloat(e.target.value);
                        stepsManager.updateStepParams(index, params);
                        
                        // 如果有图像，则重新处理
                        if (currentImageFile) {
                            scheduleProcessing();
                        }
                    });
                    
                    sliderContainer.appendChild(slider);
                    sliderContainer.appendChild(inputNumber);
                    paramDiv.appendChild(sliderContainer);
                }
                
                paramsDiv.appendChild(paramDiv);
            }
            
            contentDiv.appendChild(paramsDiv);
            stepElement.appendChild(contentDiv);
            
            // 添加折叠/展开功能
            headerDiv.addEventListener('click', (e) => {
                if (!e.target.classList.contains('delete-step-btn')) {
                    contentDiv.classList.toggle('collapsed');
                    toggleIcon.style.transform = contentDiv.classList.contains('collapsed') ? 'rotate(-90deg)' : '';
                }
            });
            
            // 如果不是第一次添加的步骤，默认折叠
            if (index > 0) {
                contentDiv.classList.add('collapsed');
                toggleIcon.style.transform = 'rotate(-90deg)';
            }
            
            elements.processingStepsList.appendChild(stepElement);
        });
        
        // 实现拖放排序
        setupDragAndDrop();
    }
    
    // 设置拖放排序功能
    function setupDragAndDrop() {
        const stepElements = document.querySelectorAll('.processing-step');
        
        stepElements.forEach(stepElement => {
            const dragHandle = stepElement.querySelector('.drag-handle');
            
            dragHandle.addEventListener('mousedown', (e) => {
                e.preventDefault();
                
                const currentStep = stepElement;
                const originalRect = currentStep.getBoundingClientRect();
                const shiftX = e.clientX - originalRect.left;
                const listRect = elements.processingStepsList.getBoundingClientRect();
                
                // 创建拖动中的克隆元素
                const clone = currentStep.cloneNode(true);
                clone.style.position = 'absolute';
                clone.style.zIndex = 1000;
                clone.style.width = `${originalRect.width}px`;
                clone.style.opacity = '0.8';
                clone.classList.add('dragging');
                
                document.body.append(clone);
                moveAt(e.pageX, e.pageY);
                
                function moveAt(pageX, pageY) {
                    const x = pageX - shiftX - listRect.left;
                    const y = pageY - originalRect.top - listRect.top;
                    
                    // 限制在列表区域内
                    const maxY = listRect.height - originalRect.height;
                    const boundedY = Math.max(0, Math.min(y, maxY));
                    
                    clone.style.left = `${listRect.left + x}px`;
                    clone.style.top = `${listRect.top + boundedY}px`;
                }
                
                function onMouseMove(e) {
                    moveAt(e.pageX, e.pageY);
                    
                    // 判断位置并调整步骤顺序
                    const targetIndex = getTargetIndex(e.clientY);
                    if (targetIndex !== -1) {
                        const fromIndex = parseInt(currentStep.dataset.index);
                        if (fromIndex !== targetIndex) {
                            stepsManager.moveStep(fromIndex, targetIndex);
                            renderProcessingSteps();
                            
                            // 如果有图像，则重新处理
                            if (currentImageFile) {
                                scheduleProcessing();
                            }
                        }
                    }
                }
                
                function getTargetIndex(clientY) {
                    let targetIndex = -1;
                    const steps = document.querySelectorAll('.processing-step');
                    
                    steps.forEach((step) => {
                        const rect = step.getBoundingClientRect();
                        if (clientY > rect.top && clientY < rect.bottom) {
                            targetIndex = parseInt(step.dataset.index);
                        }
                    });
                    
                    return targetIndex;
                }
                
                function onMouseUp() {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                    clone.remove();
                }
                
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });
        });
    }
    
    // 处理文件上传
    function handleFileUpload(file) {
        if (file) {
            currentImageFile = file;
            
            // 显示原始图像
            displayOriginalImage(file);
            
            // 显示结果区域
            elements.resultsSection.style.display = 'block';
            
            // 立即处理图像
            processImage(file);
        }
    }
    
    // 显示原始图像
    function displayOriginalImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    // 防抖函数：调度图像处理，避免频繁处理
    function scheduleProcessing() {
        if (processingTimeout) {
            clearTimeout(processingTimeout);
        }
        
        processingTimeout = setTimeout(() => {
            processImage(currentImageFile);
        }, PROCESSING_DELAY);
    }
    
    // 添加新步骤
    function addNewStep() {
        const methodId = elements.availableMethodsSelect.value;
        if (!methodId) return;
        
        stepsManager.addStep(methodId);
        renderProcessingSteps();
        
        // 如果有图像，则重新处理
        if (currentImageFile) {
            scheduleProcessing();
        }
    }
    
    // 保存模板
    function saveTemplate() {
        const templateName = elements.templateNameInput.value.trim();
        if (!templateName) {
            alert('请输入模板名称');
            return;
        }
        
        const steps = stepsManager.getSteps();
        if (steps.length === 0) {
            alert('请至少添加一个处理步骤');
            return;
        }
        
        fetch('/save_template', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                template_name: templateName,
                steps: steps
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('模板保存成功');
                loadTemplatesList();
            } else {
                alert(`保存失败: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('保存模板出错:', error);
            alert('保存模板时发生错误');
        });
    }
    
    // 加载模板列表
    function loadTemplatesList() {
        fetch('/list_templates')
        .then(response => response.json())
        .then(data => {
            const templatesList = document.getElementById('templates-list');
            templatesList.innerHTML = '';
            
            if (data.templates.length === 0) {
                templatesList.innerHTML = '<div class="empty-message">暂无保存的模板</div>';
                return;
            }
            
            data.templates.forEach(template => {
                const templateItem = document.createElement('div');
                templateItem.classList.add('template-item');
                
                const templateName = document.createElement('div');
                templateName.classList.add('template-name');
                templateName.textContent = template.replace('.json', '');
                templateItem.appendChild(templateName);
                
                const buttonsDiv = document.createElement('div');
                buttonsDiv.classList.add('template-buttons');
                
                const loadBtn = document.createElement('button');
                loadBtn.classList.add('load-template-btn');
                loadBtn.textContent = '加载';
                loadBtn.addEventListener('click', () => loadTemplate(template));
                
                const deleteBtn = document.createElement('button');
                deleteBtn.classList.add('delete-template-btn');
                deleteBtn.textContent = '删除';
                deleteBtn.addEventListener('click', () => deleteTemplate(template));
                
                buttonsDiv.appendChild(loadBtn);
                buttonsDiv.appendChild(deleteBtn);
                templateItem.appendChild(buttonsDiv);
                
                templatesList.appendChild(templateItem);
            });
        })
        .catch(error => {
            console.error('加载模板列表出错:', error);
        });
    }
    
    // 加载模板
    function loadTemplate(templateName) {
        if (!templateName) return;
        
        fetch(`/load_template/${templateName}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                stepsManager.loadTemplate(data.steps);
                renderProcessingSteps();
                alert('模板加载成功');
                
                // 如果有图像，则重新处理
                if (currentImageFile) {
                    scheduleProcessing();
                }
            } else {
                alert(`加载失败: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('加载模板出错:', error);
            alert('加载模板时发生错误');
        });
    }
    
    // 删除模板
    function deleteTemplate(templateName) {
        if (!confirm(`确定要删除模板"${templateName.replace('.json', '')}"吗？`)) {
            return;
        }
        
        fetch('/delete_template', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ template_name: templateName })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('模板删除成功');
                loadTemplatesList();
            } else {
                alert(`删除失败: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('删除模板出错:', error);
            alert('删除模板时发生错误');
        });
    }
    
    // 处理图像
    function processImage(file) {
        if (!file || isProcessing) return;
        
        isProcessing = true;
        
        // 显示加载状态
        document.body.classList.add('processing');
        
        const steps = stepsManager.getSteps();
        const formData = new FormData();
        formData.append('image', file);
        
        // 添加处理步骤到表单数据
        if (steps.length > 0) {
            formData.append('processing_steps', JSON.stringify(steps));
        }
        
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                elements.processedImage.src = data.processed_image;
                elements.downloadBtn.href = data.processed_image;
            } else {
                alert(`处理失败: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('处理图像出错:', error);
            alert('处理图像时发生错误');
        })
        .finally(() => {
            isProcessing = false;
            document.body.classList.remove('processing');
        });
    }
    
    // 初始化上传区域
    function initializeUploadArea() {
        elements.uploadArea.addEventListener('click', () => {
            elements.imageFile.click();
        });
        
        // 文件选择处理
        elements.imageFile.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFileUpload(e.target.files[0]);
            }
        });
        
        // 拖放功能
        elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.uploadArea.classList.add('dragover');
        });
        
        elements.uploadArea.addEventListener('dragleave', () => {
            elements.uploadArea.classList.remove('dragover');
        });
        
        elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
        
        // 粘贴功能
        document.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            
            for (const item of items) {
                if (item.type.indexOf('image') !== -1) {
                    const file = item.getAsFile();
                    handleFileUpload(file);
                    break;
                }
            }
        });
    }
    
    // 事件绑定
    function bindEvents() {
        // 初始化上传区域
        initializeUploadArea();
        
        // 添加步骤按钮
        elements.addStepBtn.addEventListener('click', addNewStep);
        
        // 保存模板按钮
        elements.saveTemplateBtn.addEventListener('click', saveTemplate);
        
        // 加载模板按钮
        elements.loadTemplateBtn.addEventListener('click', loadTemplate);
    }
    
    // 初始化
    function initialize() {
        initializeAvailableMethods();
        renderProcessingSteps();
        loadTemplatesList();
        bindEvents();
    }
    
    // 启动应用
    initialize();
}); 