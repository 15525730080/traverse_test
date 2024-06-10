import json
import os
import random
import sys
import time
import traceback
import webbrowser
import concurrent.futures
from datetime import datetime

from PIL import Image, ImageDraw
from jinja2 import Environment
from playwright.sync_api import sync_playwright

from image_infer import get_ui_infer

__author__ = "fanbozhou"
__email__ = "15525730080@163.com"

template_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>测试报告</title>
    <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        img {
            max-width: 100%; /* 限制图片宽度 */
            height: auto; /* 保持图片的宽高比 */
        }
    </style>
</head>
<body>
    <h1>测试报告</h1>
    <table>
        <tr>
            <th>操作元素</th>
            <th>操作前图片</th>
            <th>操作后图片</th>
        </tr>
        {% for operation in operations %}
        <tr>
            <td>{{ operation.info }}</td>
            <td><img src="{{ operation.before_image }}" alt="操作前图片"></td>
            <td><img src="{{ operation.after_image }}" alt="操作后图片"></td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


class TraverseTest(object):
    def __init__(self, target_url, output_dir):
        self.target_url = target_url
        self.ele_list = []
        self.operations = []
        self.tmp_path = None
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_image_touchability(self, image_path):
        """发送图像到服务并获取响应"""
        return get_ui_infer(image_path)

    def open_target_web_and_capture(self):
        """打开目标网页，捕获屏幕截图并上传"""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.target_url)
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except:
                pass
            self.tmp_path = "{0}.png".format(random.randint(0, 999999))
            img_bytes = page.screenshot(path=self.tmp_path)
            page.close()
            resp = self.get_image_touchability(self.tmp_path)
            print(json.dumps(resp))
            self.ele_list = resp
        self.traverse_touchable()

    def gen_report(self):
        # 设置Jinja2环境
        env = Environment()
        template = env.from_string(template_string)
        # 渲染模板
        output_html = template.render(operations=self.operations)
        # 将渲染后的HTML保存到文件
        with open("test_report.html", "w", encoding="utf-8") as f:
            f.write(output_html)

    def copy_and_mark(self, input_image_path, output_image_path, point):
        # 打开原始图片
        image = Image.open(input_image_path)
        # 复制图片
        image_copy = image.copy()

        # 标记点位
        draw = ImageDraw.Draw(image_copy)
        # 标记点位，增加线条宽度和改变颜色为红色
        draw.line((point[0] - 20, point[1], point[0] + 20, point[1]), fill=(255, 0, 0), width=10)  # 垂直
        draw.line((point[0], point[1] - 20, point[0], point[1] + 20), fill=(255, 0, 0), width=10)  # 水平

        # 保存新图片
        image_copy.save(output_image_path)

    def traverse_touchable(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            def run(index, i):
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        context = browser.new_context()
                        page = context.new_page()
                        page.goto(self.target_url)
                        try:
                            page.wait_for_load_state("networkidle", timeout=30000)
                        except:
                            pass
                        x = (i.get("elem_det_region")[0] + i.get("elem_det_region")[2]) / 2
                        y = (i.get("elem_det_region")[1] + i.get("elem_det_region")[3]) / 2
                        self.copy_and_mark(self.tmp_path, "{1}/tmp_{0}_a.png".format(index, self.output_dir), (x, y))
                        print(index, i)
                        page.mouse.click(x, y)
                        try:
                            page = page.wait_for_event("popup", timeout=10000)
                        except:
                            pass
                        time.sleep(5)
                        page.screenshot(path="{1}/tmp_{0}_b.png".format(index, self.output_dir))
                        page.close()
                        self.operations.append(
                            {"info": "click: {0}".format(i.get("elem_det_type")),
                             "before_image": "{1}/tmp_{0}_a.png".format(index, self.output_dir),
                             "after_image": "{1}/tmp_{0}_b.png".format(index, self.output_dir)}, )

                except:
                    traceback.print_exc()

            tasks = [executor.submit(run, index, i) for index, i in enumerate(self.ele_list)]
            done, not_done = concurrent.futures.wait(tasks)
            try:
                os.remove(self.tmp_path)
            except:
                traceback.print_exc()


if __name__ == "__main__":
    target_url = "https://zhaopin.meituan.com/web/social"
    try:
        if sys.argv[1]:
            target_url = sys.argv[1]
    except:
        pass
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Hh%Mm%Ss')
    output_task_path = "task_{0}".format(formatted_time)
    print("target_url: {0}".format(target_url))
    print("output_task_path: {0}".format(output_task_path))
    traverse_test = TraverseTest(
        target_url=target_url,
        output_dir=output_task_path
    )
    traverse_test.open_target_web_and_capture()
    traverse_test.gen_report()
    webbrowser.open("file://" + os.path.abspath("test_report.html"))
