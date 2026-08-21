"""
大象识别Web应用
使用Streamlit构建简单的Web界面
运行方式: streamlit run web_app.py
"""

try:
    import streamlit as st
    from PIL import Image
    import torch
    from predict import ElephantClassifier
    import cv2
    import numpy as np
    from pathlib import Path
    import json
    
    # 页面配置
    st.set_page_config(
        page_title="大象个体识别系统",
        page_icon="🐘",
        layout="wide"
    )
    
    # 标题
    st.title("🐘 大象个体识别系统")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.title("系统设置")
    
    # 检查模型是否存在
    model_path = 'best_elephant_model.pth'
    class_names_path = 'class_names.json'
    
    if not Path(model_path).exists():
        st.error("❌ 找不到训练好的模型！")
        st.info("请先运行 `python train.py` 训练模型")
        st.stop()
    
    # 加载模型（使用缓存）
    @st.cache_resource
    def load_model():
        return ElephantClassifier(model_path, class_names_path)
    
    try:
        classifier = load_model()
        st.sidebar.success("✓ 模型已加载")
        
        # 显示模型信息
        with st.sidebar.expander("模型信息"):
            st.write(f"可识别的大象: {len(classifier.class_names)}头")
            for i, name in enumerate(classifier.class_names, 1):
                st.write(f"{i}. {name}")
    
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()
    
    # 主界面标签
    tab1, tab2, tab3, tab4 = st.tabs(["📷 图片识别", "🎬 视频跟踪", "📊 数据集统计", "ℹ️ 使用说明"])
    
    # Tab 1: 图片识别
    with tab1:
        st.header("上传图片进行识别")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "选择大象图片",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="支持JPG, PNG, BMP格式"
            )
            
            if uploaded_file is not None:
                # 显示上传的图片
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的图片", use_column_width=True)
                
                # 识别按钮
                if st.button("🔍 开始识别", type="primary"):
                    with st.spinner("正在识别..."):
                        # 保存临时文件
                        temp_path = "temp_image.jpg"
                        image.save(temp_path)
                        
                        # 预测
                        elephant_name, confidence, all_probs = classifier.predict(temp_path)
                        
                        # 显示结果
                        with col2:
                            st.success("识别完成！")
                            
                            # 主要结果
                            st.metric(
                                label="识别结果",
                                value=elephant_name,
                                delta=f"{confidence:.1f}% 置信度"
                            )
                            
                            # 置信度指示
                            if confidence > 80:
                                st.success("🟢 高置信度")
                            elif confidence > 60:
                                st.warning("🟡 中等置信度")
                            else:
                                st.error("🔴 低置信度")
                            
                            # 所有类别概率
                            st.subheader("各类别概率分布")
                            
                            # 排序并显示
                            sorted_probs = sorted(all_probs.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True)
                            
                            for name, prob in sorted_probs:
                                st.progress(
                                    prob / 100,
                                    text=f"{name}: {prob:.2f}%"
                                )
    
    # Tab 2: 视频跟踪
    with tab2:
        st.header("上传视频进行大象跟踪")
        
        st.info("📌 提示: 视频处理需要较长时间，建议先用短视频（10-30秒）测试")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_video = st.file_uploader(
                "选择视频文件",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="支持MP4, AVI, MOV, MKV格式"
            )
            
            if uploaded_video is not None:
                # 保存上传的视频
                input_video_path = "temp_input_video.mp4"
                with open(input_video_path, "wb") as f:
                    f.write(uploaded_video.read())
                
                st.success(f"✓ 视频已上传: {uploaded_video.name}")
                
                # 显示视频信息
                cap = cv2.VideoCapture(input_video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                st.write(f"**视频信息:**")
                st.write(f"- 分辨率: {width}x{height}")
                st.write(f"- 帧率: {fps} FPS")
                st.write(f"- 总帧数: {frame_count}")
                st.write(f"- 时长: {duration:.1f} 秒")
                
                # 显示原视频
                st.video(input_video_path)
                
                # 处理选项
                use_yolo = st.checkbox("使用YOLO检测（更准确，需安装ultralytics）", value=True)
                show_vid_conf = st.checkbox("在输出视频中显示置信度百分比", value=False)
                freeze_locked = st.checkbox(
                    "身份确认后冻结名字（推荐，同一跟踪轨迹不易跳名）",
                    value=True,
                )
                
                # 处理按钮
                if st.button("🚀 开始跟踪识别", type="primary"):
                    output_video_path = "temp_output_video.mp4"
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # 导入跟踪器
                        from video_tracker_yolo import YOLOElephantTracker
                        
                        status_text.text("正在初始化跟踪器...")
                        
                        # 创建跟踪器
                        tracker = YOLOElephantTracker(
                            model_path='best_elephant_model.pth',
                            class_names_path='class_names.json',
                            use_yolo=use_yolo,
                            yolo_weights='yolov8m.pt',
                            show_overlay_confidence=show_vid_conf,
                            freeze_recognition_when_locked=freeze_locked,
                        )
                        
                        status_text.text("正在处理视频...")
                        
                        # 处理视频（不显示实时画面）
                        tracker.process_video(
                            input_video_path,
                            output_video_path,
                            show_live=False
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("处理完成！")
                        
                        # 在第二列显示结果
                        with col2:
                            st.success("✓ 处理完成！")
                            
                            st.write(f"**处理结果:**")
                            st.write(f"- 识别到: {len(tracker.trackers)} 头大象")
                            st.write(f"- 处理帧数: {tracker.frame_count}")
                            
                            # 列出识别到的大象
                            if tracker.trackers:
                                st.write("**识别到的大象:**")
                                for track_id, info in tracker.trackers.items():
                                    nm = info.get('name') or '未识别'
                                    st.write(f"• {nm} (置信度: {info.get('confidence', 0):.1f}%)")
                            
                            # 显示处理后的视频
                            st.video(output_video_path)
                            
                            # 下载按钮
                            with open(output_video_path, 'rb') as f:
                                st.download_button(
                                    label="📥 下载处理后的视频",
                                    data=f,
                                    file_name="elephant_tracked.mp4",
                                    mime="video/mp4"
                                )
                    
                    except ImportError as ie:
                        st.error("未安装必要的库！")
                        st.code("pip install ultralytics")
                        st.info("或者取消勾选'使用YOLO检测'")
                    except Exception as e:
                        st.error(f"处理失败: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            if uploaded_video is None:
                st.markdown("""
                ### 使用说明
                
                1. **上传视频**: 点击左侧上传按钮选择视频
                2. **选择模式**: 
                   - ✅ 使用YOLO: 更准确，需安装ultralytics
                   - ❌ 不使用YOLO: 使用背景分割，不需额外安装
                3. **开始处理**: 点击"开始跟踪识别"按钮
                4. **查看结果**: 处理完成后可在线观看或下载
                
                ### 功能特点
                
                - ✅ 自动检测和跟踪大象
                - ✅ 实时识别大象身份
                - ✅ 彩色边框标注（每头大象不同颜色）
                - ✅ 显示大象名字和置信度
                - ✅ 跟踪ID，框跟随大象移动
                
                ### YOLO安装（可选）
                
                如需更好的检测效果，请安装:
                
                ```bash
                pip install ultralytics
                ```
                
                ### 注意事项
                
                - 视频越长处理时间越久
                - 建议先用10-30秒的短视频测试
                - 确保视频中大象清晰可见
                - 处理大视频可能需要5-10分钟
                """)
    
    # Tab 3: 数据集统计
    with tab3:
        st.header("数据集统计信息")
        
        root_dir = Path('.')
        elephant_folders = sorted([d for d in root_dir.iterdir() 
                                  if d.is_dir() and not d.name.startswith('.')])
        
        if elephant_folders:
            # 统计每个类别的图片数量
            stats = {}
            total_images = 0
            
            for folder in elephant_folders:
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', 
                                   '.JPG', '.JPEG', '.PNG', '.BMP']
                images = [f for f in folder.iterdir() if f.suffix in image_extensions]
                count = len(images)
                stats[folder.name] = count
                total_images += count
            
            # 显示总览
            col1, col2, col3 = st.columns(3)
            col1.metric("大象类别数", len(stats))
            col2.metric("总图片数", total_images)
            col3.metric("平均每类", total_images // len(stats) if stats else 0)
            
            # 详细统计表格
            st.subheader("各类别详情")
            
            import pandas as pd
            df = pd.DataFrame([
                {"大象名称": name, "图片数量": count, "占比": f"{count/total_images*100:.1f}%"}
                for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)
            ])
            
            st.dataframe(df, use_container_width=True)
            
            # 柱状图
            st.subheader("图片数量分布")
            st.bar_chart(stats)
        else:
            st.warning("未找到数据集文件夹")
    
    # Tab 4: 使用说明
    with tab4:
        st.header("使用说明")
        
        st.markdown("""
        ### 📖 如何使用本系统
        
        #### 1. 图片识别
        - 点击"图片识别"标签
        - 上传一张大象图片
        - 点击"开始识别"按钮
        - 查看识别结果和置信度
        
        #### 2. 视频跟踪
        - 点击"视频跟踪"标签
        - 上传视频文件
        - 选择是否使用YOLO
        - 点击"开始跟踪识别"
        - 等待处理完成
        - 查看或下载结果视频
        
        #### 3. 理解结果
        
        **置信度说明:**
        - 🟢 **80%以上**: 高置信度，识别结果可靠
        - 🟡 **60-80%**: 中等置信度，建议人工确认
        - 🔴 **60%以下**: 低置信度，可能识别错误
        
        **视频跟踪效果:**
        - 每头大象有独特的彩色边框
        - 显示大象名字和置信度百分比
        - 框会跟随大象移动
        - 显示跟踪ID编号
        
        #### 4. YOLO vs 背景分割
        
        **使用YOLO (推荐):**
        - ✅ 检测更准确
        - ✅ 跟踪更稳定
        - ✅ 适合复杂场景
        - ❌ 需要安装ultralytics库
        
        **背景分割:**
        - ✅ 无需额外安装
        - ✅ 处理速度快
        - ❌ 适合简单场景
        - ❌ 需要静止背景
        
        #### 5. 安装YOLO（可选）
        
        在命令行中运行:
        ```bash
        pip install ultralytics
        ```
        
        #### 6. 命令行使用
        
        除了Web界面，您也可以使用命令行：
        
        ```bash
        # 图片识别
        python predict.py --mode image --image "路径/图片.jpg"
        
        # 视频跟踪
        python video_tracker_yolo.py --mode video --input "视频.mp4" --output "结果.mp4"
        
        # 摄像头
        python video_tracker_yolo.py --mode webcam
        ```
        
        #### 7. 性能优化建议
        
        - 使用GPU可以加速5-10倍
        - 先用短视频测试效果
        - 处理长视频时耐心等待
        - 确保视频质量良好
        
        ---
        
        ### 🔧 故障排除
        
        **Q: YOLO安装失败？**
        
        A: 取消勾选"使用YOLO检测"，使用背景分割模式
        
        **Q: 视频处理很慢？**
        
        A: 
        1. CPU模式较慢是正常的
        2. 建议处理短视频
        3. 考虑使用GPU版本PyTorch
        
        **Q: 识别不准确？**
        
        A:
        1. 确保图片/视频清晰
        2. 检查大象是否占据足够画面
        3. 避免严重遮挡和模糊
        
        ---
        
        **祝您使用愉快！** 🐘
        """)
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>🐘 大象个体识别系统 | "
        "基于深度学习的智能识别</div>",
        unsafe_allow_html=True
    )

except ImportError as e:
    print("="*60)
    print("错误: 缺少必要的依赖库")
    print("="*60)
    print(f"\n{e}\n")
    print("请安装Streamlit:")
    print("  pip install streamlit")
    print("\n然后运行:")
    print("  streamlit run web_app.py")
    print("="*60)
