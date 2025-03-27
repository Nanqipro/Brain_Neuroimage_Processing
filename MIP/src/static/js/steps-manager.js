/**
 * 处理步骤管理器
 * 负责管理图像处理步骤，提供添加、删除、排序等功能
 */
class ProcessingStepsManager {
    constructor() {
        this.steps = [];
        this.supportedMethods = [
            {
                id: 'apply_contrast_enhancement',
                name: '对比度增强',
                description: '通过自适应直方图均衡增强图像对比度，突出细节',
                params: {
                    clip_limit: {
                        name: '对比度限制',
                        description: '控制对比度增强的强度，值越大对比度越高',
                        type: 'number',
                        min: 0.5,
                        max: 5.0,
                        step: 0.1,
                        default: 2.0
                    }
                }
            },
            {
                id: 'edge_preserving_smooth',
                name: '边缘保持平滑',
                description: '平滑图像但保留边缘细节，比高斯模糊更好地保留图像结构',
                params: {
                    sigma_s: {
                        name: '空间系数',
                        description: '控制空间滤波范围，值越大平滑效果越强',
                        type: 'number',
                        min: 10,
                        max: 100,
                        step: 5,
                        default: 60
                    },
                    sigma_r: {
                        name: '范围系数',
                        description: '控制边缘保持程度，值越小边缘越清晰',
                        type: 'number',
                        min: 0.1,
                        max: 1.0,
                        step: 0.1,
                        default: 0.4
                    }
                }
            },
            {
                id: 'apply_discretize',
                name: '亮度离散化',
                description: '将图像亮度分成几个离散的层级，创建类似海报化或色阶分明的效果',
                params: {
                    levels: {
                        name: '层级数量',
                        description: '亮度被分成的层级数量，值越小层次感越强',
                        type: 'number',
                        min: 2,
                        max: 16,
                        step: 1,
                        default: 4
                    },
                    preserve_bright: {
                        name: '保留亮区',
                        description: '是否保留高亮区域的原始亮度',
                        type: 'select',
                        options: [
                            { value: 'false', text: '否' },
                            { value: 'true', text: '是' }
                        ],
                        default: 'true'
                    },
                    bright_threshold: {
                        name: '亮度阈值',
                        description: '高于此阈值的像素将被保留原始亮度(如果开启保留亮区)',
                        type: 'number',
                        min: 150,
                        max: 250,
                        step: 5,
                        default: 200
                    }
                }
            },
            {
                id: 'enhance_large_spots',
                name: '增强大亮块',
                description: '专门增强图像中大面积的亮区域，忽略小亮点',
                params: {
                    min_size: {
                        name: '最小尺寸',
                        description: '要增强的亮块的最小尺寸，小于此尺寸的亮点会被忽略',
                        type: 'number',
                        min: 7,
                        max: 31,
                        step: 2,
                        default: 15
                    },
                    enhancement_factor: {
                        name: '增强系数',
                        description: '控制亮块增强的强度，值越大增强效果越明显',
                        type: 'number',
                        min: 0.5,
                        max: 3.0,
                        step: 0.1,
                        default: 1.5
                    }
                }
            },
            {
                id: 'suppress_small_spots',
                name: '抑制小亮点',
                description: '减弱图像中的小亮点，保留大结构，用于去除噪点',
                params: {
                    spot_size: {
                        name: '亮点尺寸',
                        description: '要抑制的亮点的最大尺寸，大于此尺寸的亮块不受影响',
                        type: 'number',
                        min: 1,
                        max: 9,
                        step: 1,
                        default: 3
                    },
                    strength: {
                        name: '抑制强度',
                        description: '控制抑制强度，值越大小亮点越暗',
                        type: 'number',
                        min: 0.1,
                        max: 1.0,
                        step: 0.1,
                        default: 0.8
                    }
                }
            },
            {
                id: 'enhance_bright_spots',
                name: '增强亮点',
                description: '增强图像中的所有亮点区域，突出病变或特殊结构',
                params: {
                    kernel_size: {
                        name: '核大小',
                        description: '形态学操作的核大小，值越大增强范围越大',
                        type: 'number',
                        min: 3,
                        max: 15,
                        step: 2,
                        default: 5
                    },
                    enhancement_factor: {
                        name: '增强系数',
                        description: '控制亮点增强的强度，值越大增强效果越明显',
                        type: 'number',
                        min: 0.5,
                        max: 3.0,
                        step: 0.1,
                        default: 1.5
                    }
                }
            },
            {
                id: 'remove_small_spots',
                name: '去除小噪点',
                description: '通过形态学操作去除图像中的小噪点，保留主要结构',
                params: {
                    kernel_size: {
                        name: '核大小',
                        description: '形态学操作的核大小，大于此尺寸的结构会被保留',
                        type: 'number',
                        min: 1,
                        max: 9,
                        step: 1,
                        default: 3
                    },
                    iterations: {
                        name: '迭代次数',
                        description: '形态学操作的重复次数，值越大效果越强',
                        type: 'number',
                        min: 1,
                        max: 5,
                        step: 1,
                        default: 1
                    }
                }
            },
            {
                id: 'apply_bilateral_filter',
                name: '双边滤波',
                description: '在降噪的同时保持边缘锐利，适合医学图像降噪',
                params: {
                    d: {
                        name: '直径',
                        description: '每个像素邻域的直径，值越大处理越慢',
                        type: 'number',
                        min: 5,
                        max: 25,
                        step: 2,
                        default: 9
                    },
                    sigma_color: {
                        name: '颜色标准差',
                        description: '控制颜色空间的滤波强度',
                        type: 'number',
                        min: 10,
                        max: 150,
                        step: 5,
                        default: 75
                    }
                }
            },
            {
                id: 'apply_unsharp_masking',
                name: '锐化增强',
                description: '通过非锐化掩蔽技术增强图像细节和边缘',
                params: {
                    sigma: {
                        name: '模糊半径',
                        description: '控制锐化的细节范围，值越大影响范围越广',
                        type: 'number',
                        min: 0.1,
                        max: 3.0,
                        step: 0.1,
                        default: 1.0
                    },
                    amount: {
                        name: '锐化强度',
                        description: '控制锐化的强度，值越大效果越明显',
                        type: 'number',
                        min: 0.1,
                        max: 3.0,
                        step: 0.1,
                        default: 1.0
                    }
                }
            },
            {
                id: 'adjust_gamma',
                name: '伽马校正',
                description: '调整图像的伽马值，改变图像的明暗对比度',
                params: {
                    gamma: {
                        name: '伽马值',
                        description: '小于1使图像变亮，大于1使图像变暗',
                        type: 'number',
                        min: 0.1,
                        max: 3.0,
                        step: 0.1,
                        default: 1.2
                    }
                }
            },
            {
                id: 'adaptive_bright_spot_enhancement',
                name: '自适应亮块增强',
                description: '根据局部区域亮度自适应地增强亮块，更智能地处理不同区域的亮度差异',
                params: {
                    window_size: {
                        name: '窗口大小',
                        description: '局部区域分析窗口的大小，值越大考虑的范围越大',
                        type: 'number',
                        min: 3,
                        max: 31,
                        step: 2,
                        default: 15
                    },
                    sensitivity: {
                        name: '灵敏度',
                        description: '控制亮块检测的灵敏度，值越大越容易被判定为亮块',
                        type: 'number',
                        min: 0.5,
                        max: 2.0,
                        step: 0.1,
                        default: 1.2
                    },
                    min_contrast: {
                        name: '最小对比度',
                        description: '判定亮块的最小亮度差异阈值，低于此值的差异将被忽略',
                        type: 'number',
                        min: 5,
                        max: 30,
                        step: 1,
                        default: 10
                    }
                }
            },
            {
                id: 'adaptive_region_enhancement',
                name: '区域自适应增强',
                description: '将图像分成多个重叠的区块，对每个区块独立进行分析和增强，适合处理亮度分布不均的图像',
                params: {
                    region_size: {
                        name: '区块大小',
                        description: '分析区块的大小，值越大考虑的范围越大',
                        type: 'number',
                        min: 32,
                        max: 128,
                        step: 16,
                        default: 64
                    },
                    overlap: {
                        name: '重叠比例',
                        description: '相邻区块的重叠程度，值越大过渡越平滑',
                        type: 'number',
                        min: 0.1,
                        max: 0.9,
                        step: 0.1,
                        default: 0.5
                    },
                    enhancement_factor: {
                        name: '增强系数',
                        description: '控制区域增强的强度，值越大效果越明显',
                        type: 'number',
                        min: 0.8,
                        max: 2.0,
                        step: 0.1,
                        default: 1.3
                    }
                }
            }
        ];
    }

    /**
     * 获取所有支持的处理方法
     * @returns {Array} 支持的处理方法列表
     */
    getSupportedMethods() {
        return this.supportedMethods;
    }

    /**
     * 根据方法ID获取方法信息
     * @param {string} methodId 方法ID
     * @returns {object|null} 方法信息对象，如果未找到则返回null
     */
    getMethodInfo(methodId) {
        return this.supportedMethods.find(method => method.id === methodId) || null;
    }

    /**
     * 获取当前所有处理步骤
     * @returns {Array} 处理步骤列表
     */
    getSteps() {
        return [...this.steps];
    }

    /**
     * 添加处理步骤
     * @param {string} methodId 方法ID
     * @returns {boolean} 添加是否成功
     */
    addStep(methodId) {
        const methodInfo = this.getMethodInfo(methodId);
        if (!methodInfo) return false;

        // 创建步骤对象，使用默认参数
        const defaultParams = {};
        for (const paramKey in methodInfo.params) {
            defaultParams[paramKey] = methodInfo.params[paramKey].default;
        }

        this.steps.push({
            method: methodId,
            params: defaultParams
        });

        return true;
    }

    /**
     * 移除处理步骤
     * @param {number} index 步骤索引
     * @returns {boolean} 移除是否成功
     */
    removeStep(index) {
        if (index < 0 || index >= this.steps.length) return false;
        
        this.steps.splice(index, 1);
        return true;
    }

    /**
     * 更新步骤参数
     * @param {number} index 步骤索引
     * @param {Object} params 新参数
     * @returns {boolean} 更新是否成功
     */
    updateStepParams(index, params) {
        if (index < 0 || index >= this.steps.length) return false;
        
        // 合并参数，而不是完全替换
        this.steps[index].params = {
            ...this.steps[index].params,
            ...params
        };
        
        return true;
    }

    /**
     * 移动步骤位置
     * @param {number} fromIndex 源索引
     * @param {number} toIndex 目标索引
     * @returns {boolean} 移动是否成功
     */
    moveStep(fromIndex, toIndex) {
        if (
            fromIndex < 0 || fromIndex >= this.steps.length ||
            toIndex < 0 || toIndex >= this.steps.length ||
            fromIndex === toIndex
        ) {
            return false;
        }
        
        // 移动步骤
        const step = this.steps.splice(fromIndex, 1)[0];
        this.steps.splice(toIndex, 0, step);
        
        return true;
    }

    /**
     * 加载模板
     * @param {Array} steps 处理步骤列表
     * @returns {boolean} 加载是否成功
     */
    loadTemplate(steps) {
        if (!Array.isArray(steps)) return false;
        
        this.steps = steps.map(step => {
            // 确保步骤的方法在支持列表中
            const methodInfo = this.getMethodInfo(step.method);
            if (!methodInfo) return null;
            
            // 使用默认参数填充缺失的参数
            const defaultParams = {};
            for (const paramKey in methodInfo.params) {
                defaultParams[paramKey] = methodInfo.params[paramKey].default;
            }
            
            return {
                method: step.method,
                params: { ...defaultParams, ...step.params }
            };
        }).filter(step => step !== null);
        
        return true;
    }
}

// 导出处理步骤管理器
window.ProcessingStepsManager = ProcessingStepsManager; 
window.ProcessingStepsManager = ProcessingStepsManager; 