from tkinter import *
import tkinter as tk  # 使用Tkinter前需要先导入
from tkinter import filedialog, ttk
import tkinter.messagebox  # 这个是消息框，对话框的关键
from PIL import ImageTk
from PIL import Image
import math
import cv2
from colorization_by_custom_method.colorization_by_optimization.colorization_by_optimization import ColorizationByOptimization
import threading
import random
from tkinter.colorchooser import *


def mouse_left_button_down_event(event):
    global is_mouse_left_button_down,last_y,last_x,mouse_left_button_start_pos_x, mouse_left_button_start_pos_y
    if type == 0:
        last_x = event.x
        last_y = event.y
        is_mouse_left_button_down = True
        print("type0,mouse_left_button_down_event")
    else:
        is_mouse_left_button_down = True
        mouse_left_button_start_pos_y = event.y
        mouse_left_button_start_pos_x = event.x



def mouse_left_button_release_event(event):
    global is_mouse_left_button_down,target,recolor_area_info
    if type == 0:
        is_mouse_left_button_down = False
        print("mouse realease", event.x, event.y)
    else:
        target_image_left_top_x = math.floor(300 - target.width / 2)
        target_image_left_top_y = math.floor(300 - target.height / 2)
        if is_mouse_left_button_down:
            if abs(event.x - mouse_left_button_start_pos_x) < 10 or abs(event.y - mouse_left_button_start_pos_y) < 10:
                tk.messagebox.showinfo('提示', '选择区域过小')
                canvas.delete("recolorbox")
                is_mouse_left_button_down = False
                return

            recolor_area_info = {
                             "target_row1": mouse_left_button_start_pos_y - target_image_left_top_y,
                             "target_row2": event.y - target_image_left_top_y,
                             "target_col1": mouse_left_button_start_pos_x - target_image_left_top_x,
                             "target_col2": event.x - target_image_left_top_x}
            is_mouse_left_button_down = False

def mouse_motion(event):
    global last_y,last_x
    color = ["yellow", "red", "green", "gray", "blue", "white", "black"]
    text_var = "鼠标当前位置：x:{}, y:{}".format(event.x, event.y)
    var.set(text_var)
    if is_mouse_left_button_down:
        if not target:
            return
        if type == 0:
            changeColor(event.x,event.y)
            canvas.create_line(event.x, event.y, last_x, last_y, fill=pencil_color_str,width=pencil_radius)
            canvas.create_line
            last_x = event.x
            last_y = event.y
        else:
            draw_select_box(canvas, mouse_left_button_start_pos_x, mouse_left_button_start_pos_y, event.x, event.y,
                            "red", "recolorbox")

def draw_select_box(canvas, x1, y1, x2, y2, color, tag):
    canvas.delete(tag)
    if x1 > x2:
        temp = x1
        x1 = x2
        x2 = temp
    if y1 > y2:
        temp = y1
        y1 = y2
        y2 = temp
    # 绘制实线部分
    canvas.create_line(x1, y1, x1, y1 + 10, fill=color, tag=tag)
    canvas.create_line(x1, y1, x1 + 10, y1, fill=color, tag=tag)
    canvas.create_line(x2, y2, x2, y2 - 10, fill=color, tag=tag)
    canvas.create_line(x2, y2, x2 - 10, y2, fill=color, tag=tag)
    # 绘制虚线部分
    canvas.create_line(x1, y1, x1, y2, fill=color, dash=(4, 4), tag=tag)
    canvas.create_line(x1, y1, x2, y1, fill=color, dash=(4, 4), tag=tag)
    canvas.create_line(x2, y2, x2, y1, fill=color, dash=(4, 4), tag=tag)
    canvas.create_line(x2, y2, x1, y2, fill=color, dash=(4, 4), tag=tag)
    canvas.update()

def changeColor(x,y):
    global target
    target_image_left_top_x = math.floor(300 - target.width / 2)
    target_image_left_top_y = math.floor(300 - target.height / 2)
    relative_row = y - target_image_left_top_y
    relative_col = x - target_image_left_top_x
    optimization.manual_color(relative_row,relative_col,pencil_color)
    print("relative_row,relative_col",(relative_row,relative_col))

def img_select(type):
    global optimization
    global target, target_photo,  label_target_img, canvas  # 请注意必须通过global应用外部变量img_file
    file_path = filedialog.askopenfilename(initialdir='')
    if file_path == "":
        tk.messagebox.showinfo('提示', '用户取消选择')
        return
    temp_image = Image.open(file_path)
    target_width = 300
    target_height = 200
    if (temp_image.width / temp_image.height) > (target_width / target_height):
        target_height = int(target_width / temp_image.width * temp_image.height)
    else:
        target_width = int(target_height / temp_image.height * temp_image.width)

    print("target_width, target_height", target_width, target_height)
    target = temp_image.resize((target_width, target_height))
    target_photo = ImageTk.PhotoImage(target)
    # label_target_img.configure(image=target_photo)
    optimization = ColorizationByOptimization(target)
    canvas.create_image(300, 300, image=target_photo)
    canvas.update()

# def resize(event):
#     print("resize")
#     global pencil_radius
#     pencil_radius = scale.get()


def colorselect():
    global pencil_color,pencil_color_str
    color = askcolor()
    pencil_color = color[0]
    pencil_color_str = color[1]
    text = "当前颜色:{}".format(str(color[1]))
    var2.set(text)
    tk.messagebox.showinfo("提示", "你已选择颜色" + str(color[1]))

def changeType():
    global type
    type = 1-type

def show():
    optimization.show_marked_pic()

def colorization():
    print(optimization.manual_color_list)
    res = optimization.colorization()
    cv2.imshow("res",res)
    cv2.waitKey(0)

def recolorization():
    print(recolor_area_info)
    print(optimization.manual_color_list)
    optimization.set_recolor_area(recolor_area_info["target_row1"],recolor_area_info["target_row2"],recolor_area_info["target_col1"],recolor_area_info["target_col2"])
    res = optimization.colorization(is_recolor=True,debug_mode=True)
    cv2.imshow("res", res)
    cv2.waitKey(0)

type = 0
mouse_left_button_start_pos_x, mouse_left_button_start_pos_y = 0, 0
recolor_area_info = {}
pencil_color_str = "#000000"
pencil_color = (0, 0, 0)
# 鼠标左键是否处于按下框选状态
is_mouse_left_button_down = False
last_x = 0
last_y = 0
pencil_radius = 1
# UI生成———————————————————————————————————————————————————————————————————————————————————————————————————————————
# 生成窗口
root = Tk()
root.title("Picture Colorization")
root.minsize(1200, 1200)	   # 最小尺寸
root.maxsize(1200, 1200)    # 最大尺寸

# 生成introduction文字提示信息
introduction = Label(root, text="使用说明：图像上色实验", font=('Arial', 14))
introduction.pack()

# 选择图片
tk.Button(master=root, text='选择目标图片', command=lambda: img_select(type="target")).place(x=100, y=200)
tk.Button(master=root, text="选择画笔颜色", command=colorselect).place(x=100, y=500)
tk.Button(master=root, text="显示图片", command=show).place(x=100, y=600)
tk.Button(master=root, text="切换工具模式", command=changeType).place(x=100, y=800)
tk.Button(master=root, text="上色", command=colorization).place(x=100, y=300)
tk.Button(master=root, text="重上色", command=recolorization).place(x=100, y=400)
# 图片读入
target = Image.open("../../images/sample_target.png")

# 显示目标图片
# target_photo = ImageTk.PhotoImage(target)
# label_target_img = Label(root, image=target_photo)
# label_target_img.pack(anchor=N, side=TOP, padx=10, pady=10)

# frame = LabelFrame(root, height=60, width=150, text='调整画笔粗细')
# frame.pack(side='left', fill='none', expand=True)

# scale = tk.Scale(frame, from_=2, to= 50, resolution=1 ,orient=tk.HORIZONTAL, command=resize)
# scale.set(1)  # 设置初始值
# scale.grid(row=0,column=0)

# optimization = CustomMethod2(target)

# optimization.manually_set_marked_image(Image.open("./2.bmp"))
# optimization.colorization(False)

# 生成canvas画布
canvas = Canvas(root, width=600, height=600, background='white')
canvas.pack(anchor=CENTER, side=TOP, padx=0, pady=0)

pb = ttk.Progressbar(root, length=400, value=0, mode="indeterminate")
pb.pack(pady=10)

# 鼠标位置实时显示模块生成
x, y = 0, 0
var = StringVar()
text = "鼠标当前位置：x:{}, y:{}".format(x, y)
var.set(text)
lab = Label(root, textvariable=var)
lab.pack(anchor=S, side=RIGHT, padx=10, pady=10)
# 颜色信息 实时 显示模块生成
var2 = StringVar()

text = "当前颜色:#FFFFFF"
var2.set(text)
lab2 = Label(root, textvariable=var2)
lab2.pack(anchor=S, side=RIGHT, padx=10, pady=10)

canvas.bind("<Motion>", mouse_motion)
canvas.bind("<Button-1>", mouse_left_button_down_event)
canvas.bind("<ButtonRelease-1>", mouse_left_button_release_event)


mainloop()
