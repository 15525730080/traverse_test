# traverse_test

# 基于UI目标检测Web广度遍历测试
    1.使用playwright触达页面并截图
    2.通过ui_det_v2模型解析图片上的可操作元素
    3.在当前页面广度遍历的点击可操作元素并记录点击结果
    4.根据操作记录生成报告

# 依赖安装
    pip install -r requirements.txt
    playwright install
    
# 运行
    python traverse_test.py [替换成目标url，默认美团招聘网站]

# 运行效果
美团招聘：
![image](https://github.com/15525730080/traverse_test/assets/153100629/d012cc4a-dedd-4ef6-83b3-8a60929b8c17)
![image](https://github.com/15525730080/traverse_test/assets/153100629/5926f28c-00e3-4c48-b30e-586576418357)
百度：
![image](https://github.com/15525730080/traverse_test/assets/153100629/d332c04a-c57d-4679-b4e3-c8863b7c737b)
![image](https://github.com/15525730080/traverse_test/assets/153100629/5c3f7a6d-0355-46b0-937f-fdfc750f7611)

# 开源
项目侵权联系我【范博洲/15525730080@163.com】删除

# 项目仅仅是演示原理的Demo禁止商用！！！
