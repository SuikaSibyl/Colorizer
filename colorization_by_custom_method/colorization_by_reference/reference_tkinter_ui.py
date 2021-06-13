from tkinter import *
import tkinter as tk  # 使用Tkinter前需要先导入
from tkinter import filedialog, ttk
import tkinter.messagebox  # 这个是消息框，对话框的关键
from PIL import ImageTk
from PIL import Image
import math
import cv2
from colorization_by_custom_method.colorization_by_reference.colorization_by_reference import ColorizationByReference
import threading
import random
from tkinter.colorchooser import *


def change_type():
    global type
    type = 1 - type


def mouse_left_button_down_event(event):
    global is_mouse_left_button_down, mouse_left_button_start_pos_x, mouse_left_button_start_pos_y
    if type == 0:
        is_mouse_left_button_down = True
        mouse_left_button_start_pos_y = event.y
        mouse_left_button_start_pos_x = event.x
        print("mouse down", event.x, event.y)
    else:
        print("type1,mouse_left_button_down_event")


def mouse_left_button_release_event(event):
    if type == 0:
        print("mouse realease", event.x, event.y)
        print("source width height", source.width, source.height)
        print("target width height", target.width, target.height)
        source_image_left_top_x = math.floor(150 - source.width / 2)
        source_image_left_top_y = math.floor(200 - source.height / 2)
        target_image_left_top_x = math.floor(450 - target.width / 2)
        target_image_left_top_y = math.floor(200 - target.height / 2)
        global is_mouse_left_button_down, select_box_num, state, select_box_info_list, select_box_temp_info
        if is_mouse_left_button_down:
            if abs(event.x - mouse_left_button_start_pos_x) < 10 or abs(event.y - mouse_left_button_start_pos_y) < 10:
                tk.messagebox.showinfo('提示', '选择区域过小')
                canvas.delete("box" + str(select_box_num))
                is_mouse_left_button_down = False
                return
            if state == 0:
                select_box_temp_info = {"source_row1": mouse_left_button_start_pos_y - source_image_left_top_y, "source_row2": event.y - source_image_left_top_y,
                                        "source_col1": mouse_left_button_start_pos_x - source_image_left_top_x, "source_col2": event.x - source_image_left_top_x}
            if state == 1:
                select_box_num = select_box_num + 1
                select_box_info_list.append({"source_row1": select_box_temp_info["source_row1"], "source_row2": select_box_temp_info["source_row2"],
                                             "source_col1": select_box_temp_info["source_col1"], "source_col2": select_box_temp_info["source_col2"],
                                             "target_row1": mouse_left_button_start_pos_y - target_image_left_top_y, "target_row2": event.y - target_image_left_top_y,
                                             "target_col1": mouse_left_button_start_pos_x - target_image_left_top_x, "target_col2": event.x - target_image_left_top_x})
            state = 1 - state
            is_mouse_left_button_down = False
            print(select_box_info_list)
    else:
        print("type1,mouse_left_button_release_event")


def mouse_motion(event):
    color = ["yellow", "red", "green", "gray", "blue", "white", "black"]
    text_var = "鼠标当前位置：x:{}, y:{}".format(event.x, event.y)
    var.set(text_var)
    if is_mouse_left_button_down:
        draw_select_box(canvas, mouse_left_button_start_pos_x, mouse_left_button_start_pos_y, event.x, event.y, color[select_box_num], "box" + str(state) + str(select_box_num))
        # 依据鼠标起始点和当前位置点，绘制一个选框


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


def img_select(type):
    global source, target, source_photo, target_photo, label_source_img, label_target_img, canvas  # 请注意必须通过global应用外部变量img_file
    file_path = filedialog.askopenfilename(initialdir='')
    if file_path == "":
        tk.messagebox.showinfo('提示', '用户取消选择')
        return
    temp_image = Image.open(file_path)
    target_width = 250
    target_height = 150
    if (temp_image.width / temp_image.height) > (target_width / target_height):
        target_height = int(target_width / temp_image.width * temp_image.height)
    else:
        target_width = int(target_height / temp_image.height * temp_image.width)

    print("target_width, target_height", target_width, target_height)
    if type == "source":
        source = temp_image.resize((target_width, target_height))
        source_photo = ImageTk.PhotoImage(source)
        label_source_img.configure(image=source_photo)
        canvas.create_image(150, 200, image=source_photo)
        canvas.update()

    elif type == "target":
        target = temp_image.resize((target_width, target_height))
        target_photo = ImageTk.PhotoImage(target)
        label_target_img.configure(image=target_photo)
        canvas.create_image(450, 200, image=target_photo)
        canvas.update()


def global_colorization():
    pb.start()
    th = threading.Thread(target=real_global_color, args=())
    th.start()


def real_global_color():
    global source, target, canvas, global_result_photo, global_result_image
    c1 = ColorizationByReference(source, target)
    result = c1.global_colorization()
    global_result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # global_result_image.show()
    global_result_photo = ImageTk.PhotoImage(global_result_image)
    canvas.create_image(150, 400, image=global_result_photo)
    canvas.update()
    pb.stop()
    tk.messagebox.showinfo('成功', '上色完成')


def real_area_color():
    global source, target, canvas, area_result_photo, area_result_image, select_box_info_list
    c1 = ColorizationByReference(source, target)
    result = c1.area_interactive_colorization(select_box_info_list)
    area_result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # result_image.show()
    area_result_photo = ImageTk.PhotoImage(area_result_image)
    canvas.create_image(450, 400, image=area_result_photo)
    canvas.update()
    pb.stop()
    tk.messagebox.showinfo('成功', '上色完成')


def area_colorization():
    pb.start()
    th = threading.Thread(target=real_area_color, args=())
    th.start()


def colorselect():
    global pencil_color
    color = askcolor()
    pencil_color = color[0]
    text = "当前颜色:{}".format(str(color[1]))
    var2.set(text)
    tk.messagebox.showinfo("提示", "你已选择颜色" + str(color[1]))


# 全局变量定义
type = 0  # 0 为标注框, 1为上色
state = 0   # 标注参考图选框中的状态量 是在画参考图 / 目标图
pencil_color = (0, 0, 0)
# 鼠标左键是否处于按下框选状态
is_mouse_left_button_down = False
mouse_left_button_start_pos_x, mouse_left_button_start_pos_y = 0, 0
select_box_num = 0
select_box_temp_info = {}
select_box_info_list = []  # 内部记录的是相对距离

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
tk.Button(master=root, text='选择参考图片', command=lambda: img_select(type="source")).place(x=100, y=100)
tk.Button(master=root, text='选择目标图片', command=lambda: img_select(type="target")).place(x=100, y=200)
tk.Button(master=root, text='全局样本上色', command=global_colorization).place(x=100, y=300)
tk.Button(master=root, text='依据选框对应上色', command=area_colorization).place(x=100, y=400)
tk.Button(master=root, text="选择画笔颜色", command=colorselect).place(x=100, y=500)
tk.Button(master=root, text="切换模式", command=change_type).place(x=100, y=600)

# 图片读入
source = Image.open("../../images/sample_source.png")
target = Image.open("../../images/sample_target.png")
global_result_photo = target
global_result_image = target
area_result_photo = target
area_result_image = target
# 显示源图片
source_photo = ImageTk.PhotoImage(source)
label_source_img = Label(root, image=source_photo)
label_source_img.pack(anchor=N, side=TOP, padx=100, pady=10)

# 显示源图片
target_photo = ImageTk.PhotoImage(target)
label_target_img = Label(root, image=target_photo)
label_target_img.pack(anchor=N, side=TOP, padx=10, pady=10)

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
