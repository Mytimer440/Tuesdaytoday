# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import time
import sys
import os
import math
import re
import json
import ctypes
import urllib.request
import urllib.error
from pathlib import Path
from collections import deque

try:
    import bluetooth
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("缺少依赖", "未找到 pybluez/pybluez2。\n请先安装：pip install pybluez2")
    sys.exit(1)

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("缺少依赖", "未找到 bleak。\n请先安装：pip install bleak")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None


class ScrollableFrame(tk.Frame):
    """
    可滚动页面容器。
    新版不再对所有控件反复 bind_all / unbind_all，
    而是在鼠标进入当前页面后接管滚轮，离开后释放，滚动更稳定。
    """
    def __init__(self, master, bg="#000000", **kwargs):
        super().__init__(master, bg=bg, **kwargs)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll.pack(side="right", fill="y")

        self._mouse_inside = False

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind("<Enter>", self._activate_mousewheel)
        self.bind("<Leave>", self._deactivate_mousewheel)
        self.canvas.bind("<Enter>", self._activate_mousewheel)
        self.canvas.bind("<Leave>", self._deactivate_mousewheel)
        self.inner.bind("<Enter>", self._activate_mousewheel)
        self.inner.bind("<Leave>", self._deactivate_mousewheel)

    def refresh_child_bindings(self):
        # 兼容旧调用。新版不再递归绑定每个子控件，避免滚轮冲突。
        pass

    def _on_inner_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _activate_mousewheel(self, event=None):
        self._mouse_inside = True
        top = self.winfo_toplevel()
        top.bind_all("<MouseWheel>", self._on_mousewheel_windows_mac, add="+")
        top.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        top.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _deactivate_mousewheel(self, event=None):
        self._mouse_inside = False
        try:
            top = self.winfo_toplevel()
            top.unbind_all("<MouseWheel>")
            top.unbind_all("<Button-4>")
            top.unbind_all("<Button-5>")
        except Exception:
            pass

    def _event_belongs_here(self, event):
        if not self._mouse_inside:
            return False
        widget = getattr(event, "widget", None)
        if widget is None:
            return True
        try:
            cls = widget.winfo_class()
            if cls in {"Treeview", "Text", "TScrollbar", "Scrollbar", "Listbox"}:
                return False
        except Exception:
            pass
        return True

    def _on_mousewheel_windows_mac(self, event):
        if not self._event_belongs_here(event):
            return
        try:
            delta = event.delta
            if delta == 0:
                return "break"
            units = -1 if delta > 0 else 1
            steps = max(1, min(6, int(abs(delta) / 120) or 1))
            self.canvas.yview_scroll(units * steps * 3, "units")
            return "break"
        except Exception:
            return "break"

    def _on_mousewheel_linux(self, event):
        if not self._event_belongs_here(event):
            return
        try:
            if event.num == 4:
                self.canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(3, "units")
            return "break"
        except Exception:
            return "break"


class ReadOnlyScrolledText(scrolledtext.ScrolledText):
    """只读但允许复制/选择"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind("<Key>", self._block_edit)
        self.bind("<Control-v>", self._block_edit)
        self.bind("<Control-V>", self._block_edit)
        self.bind("<Button-2>", self._block_edit)
        self.bind("<<Paste>>", self._block_edit)
        self.bind("<<Cut>>", self._block_edit)
        self.bind("<Delete>", self._block_edit)
        self.bind("<BackSpace>", self._block_edit)

    def _block_edit(self, event):
        allowed = {
            "Left", "Right", "Up", "Down",
            "Home", "End", "Prior", "Next",
            "Shift_L", "Shift_R", "Control_L", "Control_R",
            "c", "C", "a", "A"
        }
        ctrl = (event.state & 0x4) != 0
        if ctrl and event.keysym in {"c", "C", "a", "A"}:
            return
        if event.keysym in allowed:
            return
        return "break"


class RoundedButton(tk.Canvas):
    def __init__(
        self,
        master,
        text,
        command,
        width=300,
        height=68,
        radius=22,
        bg="#245dff",
        active_bg="#3f78ff",
        fg="white",
        font=("Microsoft YaHei", 16, "bold"),
        **kwargs
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            bg=master.cget("bg"),
            **kwargs
        )
        self.command = command
        self.normal_bg = bg
        self.active_bg = active_bg
        self.current_bg = bg
        self.disabled_bg = "#233149"
        self.text_color = fg
        self.font = font
        self.radius = radius
        self.button_text = text
        self.enabled = True

        self._draw()

        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _round_rect_points(self, x1, y1, x2, y2, r):
        return [
            x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1,
            x2, y1, x2, y1+r, x2, y1+r, x2, y2-r,
            x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
            x1+r, y2, x1+r, y2, x1, y2, x1, y2-r,
            x1, y2-r, x1, y1+r, x1, y1+r, x1, y1
        ]

    def _draw(self):
        self.delete("all")
        w = int(self["width"])
        h = int(self["height"])
        pts = self._round_rect_points(2, 2, w - 2, h - 2, self.radius)
        self.create_polygon(pts, smooth=True, fill=self.current_bg, outline="")
        self.create_text(
            w // 2, h // 2,
            text=self.button_text,
            fill=self.text_color,
            font=self.font
        )

    def _on_enter(self, event):
        if self.enabled:
            self.current_bg = self.active_bg
            self._draw()

    def _on_leave(self, event):
        if self.enabled:
            self.current_bg = self.normal_bg
            self._draw()

    def _on_click(self, event):
        if self.enabled and self.command:
            self.command()

    def config_text(self, text):
        self.button_text = text
        self._draw()

    def set_enabled(self, enabled=True):
        self.enabled = enabled
        self.current_bg = self.normal_bg if enabled else self.disabled_bg
        self._draw()



class BluetoothWristbandAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("运动手环监控系统")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1680, int(screen_w * 0.92))
        win_h = min(980, int(screen_h * 0.88))
        self.root.geometry(f"{win_w}x{win_h}")
        self.root.minsize(min(980, screen_w), min(680, screen_h))
        self.root.configure(bg="#07111f")

        self.bg = "#07111f"
        self.card = "#0d1726"
        self.card2 = "#111d2e"
        self.card3 = "#15243a"
        self.text_main = "#f4f8ff"
        self.text_dim = "#9fb0c6"
        self.accent_time = "#60a5fa"
        self.accent_green = "#34d399"
        self.accent_red = "#fb7185"
        self.warning = "#fbbf24"
        self.border = "#20324d"
        self.input_bg = "#162235"
        self.btn_bg = "#1e3a5f"
        self.btn_active = "#28527f"
        self.accent_cyan = "#22d3ee"
        self.accent_yellow = "#facc15"

        self.bt_type = tk.StringVar(value="SPP")
        self.bt_port = tk.IntVar(value=1)
        self.ble_char_write_uuid = tk.StringVar(value="6e400002-b5a3-f393-e0a9-e50e24dcca9e")
        self.ble_char_notify_uuid = tk.StringVar(value="6e400003-b5a3-f393-e0a9-e50e24dcca9e")

        self.is_connected = False
        self.bt_socket = None
        self.ble_client = None
        self.stop_event = threading.Event()
        self.selected_device_addr = None

        self.ble_loop = None
        self.ble_thread = None

        self.device_connected = False

        self.current_hr = 0.0
        self.current_spo2 = 0.0

        self.x_pos = 0.0
        self.x_neg = 0.0
        self.y_pos = 0.0
        self.y_neg = 0.0
        self.z_pos = 0.0
        self.z_neg = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0

        self.position_text = "未知"
        self.rx_text_buffer = ""

        # 主界面不显示运行日志，日志写入本地文件，避免日志框影响实时刷新。
        self.log_file_path = None
        self.latest_status_text = tk.StringVar(value="系统准备就绪")
        self.dashboard_update_pending = False
        self.last_dashboard_update_time = 0.0

        # DeepSeek API 留空，由用户自行填写。
        self.deepseek_api_key = tk.StringVar(value=os.getenv("DEEPSEEK_API_KEY", ""))
        self.deepseek_model_name = tk.StringVar(value="deepseek-chat")
        self.ai_result_text_widget = None

        # 心率/血氧历史，用于弹窗折线图。
        self.hr_history = deque(maxlen=600)
        self.spo2_history = deque(maxlen=600)
        self.history_time = deque(maxlen=600)

        self.step_count = 0

        # ===== StepPeakNet-25Hz DLL 模型计步状态 =====
        # 旧版阈值峰值算法不再作为计步来源；这里改为 C++ DLL + ONNX 模型输出 + 后处理。
        self.step_model_enabled = False
        self.step_engine_dll = None
        self.step_dll_directory_cookie = None
        self.step_norm_mean = None
        self.step_norm_std = None

        self.imu_buffer = deque(maxlen=64)       # 最近 64 帧六轴数据
        self.model_prob_queue = deque(maxlen=3)  # 用 3 点局部峰值判断是否加步
        self.imu_frame_index = 0
        self.frames_since_infer = 0
        self.last_model_step_frame = -999999

        self.model_window_size = 64
        self.model_stride = 5
        self.model_step_threshold = 0.50
        self.model_gait_threshold = 0.50
        self.model_min_distance = 7
        self.app_base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        try:
            self.log_file_path = Path(__file__).resolve().parent / "wristband_runtime.log"
        except Exception:
            self.log_file_path = self.app_base_dir / "wristband_runtime.log"

        self.model_paths = {
            "onnx": self.app_base_dir / "model_run_v2" / "steppeaknet_25hz.onnx",
            "norm": self.app_base_dir / "processed_dataset" / "norm_stats.json",
            "dll_dir": self.app_base_dir / "step_engine_runtime",
            "dll": self.app_base_dir / "step_engine_runtime" / "step_engine.dll",
        }

        # ===== 实时采样率适配状态 =====
        # 模型训练时固定为 25Hz；这里把蓝牙实际收到的 22~28Hz 等不稳定数据
        # 统一线性插值成 25Hz 后，再送入原来的 64帧模型窗口。
        self.resample_enabled = True
        self.target_hz = 25.0
        self.target_dt = 1.0 / self.target_hz  # 0.04秒
        self.raw_imu_time_buffer = deque(maxlen=250)  # 原始输入缓存: (timestamp_s, [ax,ay,az,gx,gy,gz])
        self.input_time_buffer = deque(maxlen=250)    # 用于估算当前实际输入Hz
        self.next_resample_time = None
        self.max_interpolate_gap = 0.25               # 超过0.25秒认为蓝牙卡顿，不强行补帧
        self.last_resample_log_time = 0.0
        self.last_resample_reset_log_time = 0.0

        self.health_events = deque(maxlen=500)
        self.last_abnormal_record_time = {
            "spo2_low": 0,
            "spo2_critical": 0,
            "hr_low": 0,
            "hr_high": 0,
            "hr_critical_low": 0,
            "hr_critical_high": 0,
        }
        self.abnormal_record_cooldown = 30

        self.current_layout_mode = None

        self.setup_styles()
        self.create_pages()
        self.set_bt_ui_state("disconnected")
        self.init_step_model()
        self.start_ble_event_loop()

        self.root.bind("<Configure>", self.on_root_resize)

        self.log_message("系统初始化完成。", "success")
        self.log_message("请先选择协议，再点击“刷新设备列表”。", "info")
        self.log_message("健康异常时序记录功能已启用。🩺", "info")

    def setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Dark.TCombobox",
            fieldbackground=self.input_bg,
            background=self.input_bg,
            foreground=self.text_main,
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border,
            arrowsize=14,
            padding=6,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", self.input_bg)],
            foreground=[("readonly", self.text_main)],
            background=[("readonly", self.input_bg)],
        )

        style.configure(
            "Dark.TEntry",
            fieldbackground=self.input_bg,
            background=self.input_bg,
            foreground=self.text_main,
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border,
            insertcolor=self.text_main,
            padding=6,
        )

        style.configure(
            "Dark.Treeview",
            background="#0b0e13",
            fieldbackground="#0b0e13",
            foreground=self.text_main,
            rowheight=30,
            bordercolor=self.border,
            lightcolor=self.border,
            darkcolor=self.border,
            font=("Consolas", 10),
        )

        style.configure(
            "Dark.Treeview.Heading",
            background="#121722",
            foreground=self.text_main,
            relief="flat",
            font=("Microsoft YaHei", 10, "bold"),
        )

        style.map(
            "Dark.Treeview",
            background=[("selected", "#1a2433")],
            foreground=[("selected", self.text_main)],
        )

    def dark_button(self, parent, text, command, width=None, state=tk.NORMAL):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.btn_bg,
            fg=self.text_main,
            activebackground=self.btn_active,
            activeforeground=self.text_main,
            relief="flat",
            bd=0,
            font=("Microsoft YaHei", 10, "bold"),
            padx=16,
            pady=9,
            cursor="hand2",
            width=width,
            state=state,
        )

    def create_pages(self):
        self.page_container = tk.Frame(self.root, bg=self.bg)
        self.page_container.pack(fill="both", expand=True)

        self.main_page_wrapper = ScrollableFrame(self.page_container, bg=self.bg)
        self.abnormal_page_wrapper = ScrollableFrame(self.page_container, bg=self.bg)
        self.ai_page_wrapper = ScrollableFrame(self.page_container, bg=self.bg)

        self.main_page = self.main_page_wrapper.inner
        self.abnormal_page = self.abnormal_page_wrapper.inner
        self.ai_page = self.ai_page_wrapper.inner

        self.create_main_widgets(self.main_page)
        self.create_abnormal_page(self.abnormal_page)
        self.create_ai_page(self.ai_page)

        self.show_main_page()

    def _hide_all_pages(self):
        self.main_page_wrapper.pack_forget()
        self.abnormal_page_wrapper.pack_forget()
        self.ai_page_wrapper.pack_forget()

    def show_main_page(self):
        self._hide_all_pages()
        self.main_page_wrapper.pack(fill="both", expand=True)

    def show_abnormal_page(self):
        self._hide_all_pages()
        self.abnormal_page_wrapper.pack(fill="both", expand=True)
        self.refresh_abnormal_page()

    def show_ai_page(self):
        self._hide_all_pages()
        self.ai_page_wrapper.pack(fill="both", expand=True)

    def create_main_widgets(self, parent):
        outer = tk.Frame(parent, bg=self.bg)
        outer.pack(fill="both", expand=True, padx=18, pady=16)

        top_bar = tk.Frame(outer, bg=self.bg)
        top_bar.pack(fill="x", pady=(0, 14))

        self.top_time_label = tk.Label(
            top_bar,
            text=time.strftime("%H:%M:%S"),
            fg=self.accent_time,
            bg=self.bg,
            font=("Consolas", 15, "bold"),
        )
        self.top_time_label.pack(side="right")

        self.abnormal_jump_btn = self.dark_button(top_bar, "异常记录", self.show_abnormal_page, width=10)
        self.abnormal_jump_btn.pack(side="right", padx=(0, 10))

        self.ai_jump_btn = self.dark_button(top_bar, "AI健康分析", self.show_ai_page, width=12)
        self.ai_jump_btn.pack(side="right", padx=(0, 10))

        self.chart_btn = self.dark_button(top_bar, "趋势图", self.show_trend_chart_popup, width=8)
        self.chart_btn.pack(side="right", padx=(0, 10))

        self.update_top_time()

        status_strip = tk.Frame(outer, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        status_strip.pack(fill="x", pady=(0, 14))
        tk.Label(
            status_strip,
            textvariable=self.latest_status_text,
            fg=self.text_dim,
            bg=self.card,
            font=("Microsoft YaHei", 10, "bold"),
        ).pack(side="left", padx=14, pady=10)
        self.open_log_btn = self.dark_button(status_strip, "打开日志文件", self.open_log_file, width=12)
        self.open_log_btn.pack(side="right", padx=10, pady=8)

        self.main_area = tk.Frame(outer, bg=self.bg)
        self.main_area.pack(fill="both", expand=True)

        self.health_col = tk.Frame(self.main_area, bg=self.bg)
        self.control_col = tk.Frame(self.main_area, bg=self.bg)

        # ===== 左侧：健康数据卡片 =====
        health_panel = tk.Frame(self.health_col, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        health_panel.pack(fill="both", expand=True)

        health_header = tk.Frame(health_panel, bg=self.card)
        health_header.pack(fill="x", padx=18, pady=(16, 12))

        tk.Label(
            health_header,
            text="健康监测",
            fg=self.text_main,
            bg=self.card,
            font=("Microsoft YaHei", 18, "bold"),
        ).pack(side="left")

        self.btn_reset_steps = self.dark_button(health_header, "重置步数", self.reset_steps, width=10, state=tk.DISABLED)
        self.btn_reset_steps.pack(side="right")

        self.health_content = tk.Frame(health_panel, bg=self.card)
        self.health_content.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.status_card = tk.Frame(self.health_content, bg=self.card2, highlightthickness=1, highlightbackground=self.border)
        self.status_card.pack(fill="x", pady=(0, 14))

        self.status_dot = tk.Label(self.status_card, text="●", bg=self.card2, fg=self.accent_time, font=("Microsoft YaHei", 18, "bold"))
        self.status_dot.pack(side="left", padx=(16, 8), pady=14)

        self.status_text_label = tk.Label(
            self.status_card,
            text="综合状态：等待数据",
            bg=self.card2,
            fg=self.accent_time,
            font=("Microsoft YaHei", 13, "bold"),
        )
        self.status_text_label.pack(side="left", pady=14)

        self.metric_grid = tk.Frame(self.health_content, bg=self.card)
        self.metric_grid.pack(fill="x", pady=(0, 14))

        self.spo2_value_label, self.spo2_sub_label, self.spo2_progress = self.create_metric_widget(
            self.metric_grid, "血氧 SpO₂", "0%", "无数据", self.accent_green
        )
        self.hr_value_label, self.hr_sub_label, self.hr_progress = self.create_metric_widget(
            self.metric_grid, "心率 BPM", "0", "无数据", self.accent_red
        )
        self.step_value_label, self.step_sub_label, self.step_progress = self.create_metric_widget(
            self.metric_grid, "步数 Steps", "0", "今日累计", self.warning
        )

        for i in range(3):
            self.metric_grid.grid_columnconfigure(i, weight=1)

        # ===== 右侧：蓝牙连接与功能入口 =====
        bluetooth_panel = tk.Frame(self.control_col, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        bluetooth_panel.pack(fill="both", expand=True)

        conn_card = tk.Frame(bluetooth_panel, bg=self.card2, highlightthickness=1, highlightbackground=self.border)
        conn_card.pack(fill="both", expand=True, padx=16, pady=(16, 10))

        conn_header = tk.Frame(conn_card, bg=self.card2)
        conn_header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=16, pady=(14, 12))

        tk.Label(
            conn_header,
            text="蓝牙连接设置",
            fg=self.text_main,
            bg=self.card2,
            font=("Microsoft YaHei", 13, "bold"),
        ).pack(side="left")

        self.wristband_status_label = tk.Label(
            conn_header,
            text="手环状态：未连接",
            font=("Microsoft YaHei", 10, "bold"),
            bg=self.card2,
            fg=self.accent_red,
        )
        self.wristband_status_label.pack(side="right")

        tk.Label(
            conn_card,
            text="连接类型：",
            fg=self.text_dim,
            bg=self.card2,
            font=("Microsoft YaHei", 10),
        ).grid(row=1, column=0, sticky="w", padx=16, pady=8)

        tk.Label(
            conn_card,
            text="SPP 串口透传",
            fg=self.accent_green,
            bg=self.card2,
            font=("Microsoft YaHei", 11, "bold"),
        ).grid(row=1, column=1, sticky="w", padx=8, pady=8)

        tk.Label(
            conn_card,
            text="SPP端口：",
            fg=self.text_dim,
            bg=self.card2,
            font=("Microsoft YaHei", 10),
        ).grid(row=2, column=0, sticky="w", padx=16, pady=8)

        ttk.Entry(conn_card, textvariable=self.bt_port, width=8, style="Dark.TEntry").grid(row=2, column=1, sticky="w", padx=8, pady=8)

        tk.Label(
            conn_card,
            text="设备列表：",
            fg=self.text_dim,
            bg=self.card2,
            font=("Microsoft YaHei", 10),
        ).grid(row=3, column=0, sticky="nw", padx=16, pady=(8, 8))

        list_frame = tk.Frame(conn_card, bg=self.card2)
        list_frame.grid(row=3, column=1, columnspan=2, sticky="nsew", padx=(8, 16), pady=(8, 12))

        self.device_tree = ttk.Treeview(
            list_frame,
            columns=("addr", "name"),
            show="headings",
            height=10,
            style="Dark.Treeview",
        )
        self.device_tree.heading("addr", text="地址")
        self.device_tree.heading("name", text="名称")
        self.device_tree.column("addr", width=190, stretch=True)
        self.device_tree.column("name", width=220, stretch=True)
        self.device_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.device_tree.bind("<Double-1>", self.on_device_double_click)
        self.device_tree.bind("<<TreeviewSelect>>", self.on_device_select)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.device_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.device_tree.configure(yscrollcommand=scrollbar.set)

        btn_frame = tk.Frame(conn_card, bg=self.card2)
        btn_frame.grid(row=4, column=1, columnspan=2, sticky="w", padx=(8, 16), pady=(0, 14))

        self.refresh_btn = self.dark_button(btn_frame, "刷新设备列表", self.scan_devices)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.conn_btn = self.dark_button(btn_frame, "连接选中设备", self.toggle_connection, state=tk.DISABLED)
        self.conn_btn.pack(side=tk.LEFT)

        conn_card.grid_columnconfigure(1, weight=1)
        conn_card.grid_rowconfigure(3, weight=1)

        tools_card = tk.Frame(bluetooth_panel, bg=self.card2, highlightthickness=1, highlightbackground=self.border)
        tools_card.pack(fill="x", padx=16, pady=(0, 16))

        tk.Label(
            tools_card,
            text="快捷功能",
            fg=self.text_main,
            bg=self.card2,
            font=("Microsoft YaHei", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 10))

        tool_row = tk.Frame(tools_card, bg=self.card2)
        tool_row.pack(fill="x", padx=16, pady=(0, 16))

        self.ai_btn2 = self.dark_button(tool_row, "AI健康分析", self.show_ai_page, width=12)
        self.ai_btn2.pack(side="left", padx=(0, 10))

        self.trend_btn2 = self.dark_button(tool_row, "查看折线图", self.show_trend_chart_popup, width=12)
        self.trend_btn2.pack(side="left", padx=(0, 10))

        self.log_btn2 = self.dark_button(tool_row, "打开日志", self.open_log_file, width=10)
        self.log_btn2.pack(side="left", padx=(0, 10))

        self.send_time_btn = self.dark_button(tool_row, "发送电脑时间", self.send_pc_time_to_device, width=12, state=tk.DISABLED)
        self.send_time_btn.pack(side="left")

        self.apply_responsive_layout(force=True)
        self.redraw_watch_panel()

    def create_metric_widget(self, parent, title, value, subtitle, color):
        idx = len(parent.grid_slaves())
        col = idx
        frame = tk.Frame(parent, bg=self.card2, highlightthickness=1, highlightbackground=self.border)
        frame.grid(row=0, column=col, sticky="nsew", padx=6, pady=0)

        tk.Label(
            frame,
            text=title,
            bg=self.card2,
            fg=self.text_dim,
            font=("Microsoft YaHei", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(14, 8))

        value_label = tk.Label(
            frame,
            text=value,
            bg=self.card2,
            fg=color,
            font=("Segoe UI", 28, "bold"),
        )
        value_label.pack(anchor="center", pady=(2, 2))

        sub_label = tk.Label(
            frame,
            text=subtitle,
            bg=self.card2,
            fg=self.text_main,
            font=("Microsoft YaHei", 9, "bold"),
        )
        sub_label.pack(anchor="center", pady=(0, 10))

        progress = tk.Canvas(frame, height=10, bg=self.card2, highlightthickness=0, bd=0)
        progress.pack(fill="x", padx=14, pady=(0, 14))
        progress._bar_color = color
        progress._ratio = 0.0
        progress.bind("<Configure>", lambda e, c=progress: self.redraw_metric_progress(c))

        return value_label, sub_label, progress

    def redraw_metric_progress(self, canvas):
        try:
            canvas.delete("all")
            w = max(10, canvas.winfo_width())
            h = max(8, canvas.winfo_height())
            self.round_rect(canvas, 0, 0, w, h, r=5, fill="#223149", outline="")
            ratio = max(0.0, min(1.0, getattr(canvas, "_ratio", 0.0)))
            if ratio > 0:
                self.round_rect(canvas, 0, 0, w * ratio, h, r=5, fill=getattr(canvas, "_bar_color", self.accent_green), outline="")
        except Exception:
            pass

    def set_metric_progress(self, canvas, value, min_val, max_val):
        try:
            if value <= 0:
                ratio = 0.0
            else:
                ratio = (value - min_val) / max(1e-6, (max_val - min_val))
            canvas._ratio = max(0.0, min(1.0, ratio))
            self.redraw_metric_progress(canvas)
        except Exception:
            pass

    def create_abnormal_page(self, parent):
        outer = tk.Frame(parent, bg=self.bg)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        top_bar = tk.Frame(outer, bg=self.bg)
        top_bar.pack(fill="x", pady=(0, 12))

        tk.Label(
            top_bar,
            text="健康异常记录中心",
            fg=self.text_main,
            bg=self.bg,
            font=("Microsoft YaHei", 20, "bold"),
        ).pack(side="left")

        right_tools = tk.Frame(top_bar, bg=self.bg)
        right_tools.pack(side="right")

        self.abnormal_page_time = tk.Label(
            right_tools,
            text=time.strftime("%H:%M:%S"),
            fg=self.accent_time,
            bg=self.bg,
            font=("Consolas", 14, "bold"),
        )
        self.abnormal_page_time.pack(side="right")

        back_btn = self.dark_button(right_tools, "返回主页面", self.show_main_page)
        back_btn.pack(side="right", padx=(0, 12))

        summary_card = tk.Frame(outer, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        summary_card.pack(fill="x", pady=(0, 12))

        tk.Label(
            summary_card,
            text="异常统计摘要",
            fg=self.text_main,
            bg=self.card,
            font=("Microsoft YaHei", 15, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 12))

        summary_grid = tk.Frame(summary_card, bg=self.card)
        summary_grid.pack(fill="x", padx=18, pady=(0, 16))

        self.summary_total = self.create_summary_item(summary_grid, "总异常次数", "0", self.warning)
        self.summary_spo2 = self.create_summary_item(summary_grid, "血氧异常", "0", self.accent_green)
        self.summary_hr = self.create_summary_item(summary_grid, "心率异常", "0", self.accent_red)
        self.summary_critical = self.create_summary_item(summary_grid, "严重异常", "0", self.accent_yellow)
        self.summary_latest = self.create_summary_item(summary_grid, "最近异常时间", "无", self.accent_time, wide=True)

        self.summary_total["frame"].grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.summary_spo2["frame"].grid(row=0, column=1, padx=8, pady=8, sticky="nsew")
        self.summary_hr["frame"].grid(row=0, column=2, padx=8, pady=8, sticky="nsew")
        self.summary_critical["frame"].grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        self.summary_latest["frame"].grid(row=1, column=1, columnspan=2, padx=8, pady=8, sticky="nsew")

        for i in range(3):
            summary_grid.grid_columnconfigure(i, weight=1)

        list_card = tk.Frame(outer, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        list_card.pack(fill="both", expand=True)

        list_header = tk.Frame(list_card, bg=self.card)
        list_header.pack(fill="x", padx=18, pady=(16, 10))

        tk.Label(
            list_header,
            text="异常记录列表",
            fg=self.text_main,
            bg=self.card,
            font=("Microsoft YaHei", 15, "bold"),
        ).pack(side="left")

        self.abnormal_count_hint = tk.Label(
            list_header,
            text="当前 0 条记录",
            fg=self.text_dim,
            bg=self.card,
            font=("Microsoft YaHei", 10, "bold"),
        )
        self.abnormal_count_hint.pack(side="left", padx=(12, 0))

        btn_tools = tk.Frame(list_header, bg=self.card)
        btn_tools.pack(side="right")

        refresh_btn = self.dark_button(btn_tools, "刷新", self.refresh_abnormal_page, width=8)
        refresh_btn.pack(side="left", padx=(0, 10))

        clear_event_btn = self.dark_button(btn_tools, "清空异常记录", self.clear_health_events, width=12)
        clear_event_btn.pack(side="left")

        tree_wrap = tk.Frame(list_card, bg=self.card)
        tree_wrap.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.abnormal_tree = ttk.Treeview(
            tree_wrap,
            columns=("time", "type", "value", "level", "description"),
            show="headings",
            style="Dark.Treeview",
        )
        self.abnormal_tree.heading("time", text="时间")
        self.abnormal_tree.heading("type", text="异常类型")
        self.abnormal_tree.heading("value", text="数值")
        self.abnormal_tree.heading("level", text="等级")
        self.abnormal_tree.heading("description", text="描述")

        self.abnormal_tree.column("time", width=160, anchor="center", stretch=False)
        self.abnormal_tree.column("type", width=110, anchor="center", stretch=False)
        self.abnormal_tree.column("value", width=90, anchor="center", stretch=False)
        self.abnormal_tree.column("level", width=90, anchor="center", stretch=False)
        self.abnormal_tree.column("description", width=400, anchor="w", stretch=True)

        self.abnormal_tree.pack(side="left", fill="both", expand=True)

        tree_scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.abnormal_tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.abnormal_tree.configure(yscrollcommand=tree_scroll.set)

    def create_ai_page(self, parent):
        outer = tk.Frame(parent, bg=self.bg)
        outer.pack(fill="both", expand=True, padx=18, pady=16)

        top_bar = tk.Frame(outer, bg=self.bg)
        top_bar.pack(fill="x", pady=(0, 14))

        tk.Label(
            top_bar,
            text="AI 健康分析",
            fg=self.text_main,
            bg=self.bg,
            font=("Microsoft YaHei", 20, "bold"),
        ).pack(side="left")

        back_btn = self.dark_button(top_bar, "返回主页面", self.show_main_page)
        back_btn.pack(side="right")

        api_card = tk.Frame(outer, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        api_card.pack(fill="x", pady=(0, 12))

        row1 = tk.Frame(api_card, bg=self.card)
        row1.pack(fill="x", padx=18, pady=(16, 8))
        tk.Label(row1, text="DeepSeek API Key：", bg=self.card, fg=self.text_dim, font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        ttk.Entry(row1, textvariable=self.deepseek_api_key, width=54, show="*", style="Dark.TEntry").pack(side="left", padx=8)

        row2 = tk.Frame(api_card, bg=self.card)
        row2.pack(fill="x", padx=18, pady=(0, 16))
        tk.Label(row2, text="模型名称：", bg=self.card, fg=self.text_dim, font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        ttk.Entry(row2, textvariable=self.deepseek_model_name, width=24, style="Dark.TEntry").pack(side="left", padx=8)
        self.ai_analyze_btn = self.dark_button(row2, "开始分析", self.start_ai_health_analysis, width=10)
        self.ai_analyze_btn.pack(side="left", padx=(12, 0))

        result_card = tk.Frame(outer, bg=self.card, highlightthickness=1, highlightbackground=self.border)
        result_card.pack(fill="both", expand=True)

        tk.Label(
            result_card,
            text="分析结果",
            bg=self.card,
            fg=self.text_main,
            font=("Microsoft YaHei", 15, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 10))

        self.ai_result_text_widget = ReadOnlyScrolledText(
            result_card,
            bg="#0b1320",
            fg=self.text_main,
            insertbackground=self.text_main,
            relief="flat",
            bd=0,
            font=("Microsoft YaHei", 11),
            wrap=tk.WORD,
            height=18,
        )
        self.ai_result_text_widget.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.set_ai_result("请填写 DeepSeek API Key 后点击“开始分析”。\\n\\nAPI 地址默认使用：https://api.deepseek.com/chat/completions\\nAPI Key 留空由你自行填写，不会写死在程序里。")

    def set_ai_result(self, text):
        if not self.ai_result_text_widget:
            return
        try:
            self.ai_result_text_widget.config(state=tk.NORMAL)
            self.ai_result_text_widget.delete("1.0", tk.END)
            self.ai_result_text_widget.insert(tk.END, text)
            self.ai_result_text_widget.config(state=tk.DISABLED)
        except Exception:
            pass

    def build_ai_health_prompt(self):
        recent_events = self.get_recent_health_events_text(max_count=10)
        return (
            "你是一个运动手环健康监测助手。请根据以下实时数据进行中文分析，"
            "要求：先给出总体判断，再分点说明心率、血氧、步数和异常记录，最后给出温和建议。"
            "不要做医疗诊断，不要替代医生。\\n\\n"
            f"当前血氧 SpO2：{self.current_spo2:.1f}%\\n"
            f"当前心率 BPM：{self.current_hr:.1f}\\n"
            f"当前步数：{self.step_count}\\n"
            f"手环位置：{self.position_text}\\n"
            f"最近异常记录：\\n{recent_events}\\n"
        )

    def start_ai_health_analysis(self):
        api_key = self.deepseek_api_key.get().strip()
        if not api_key:
            messagebox.showwarning("缺少 API Key", "请先填写 DeepSeek API Key。")
            return

        self.set_ai_result("正在调用 DeepSeek API，请稍等……")
        threading.Thread(target=self._run_ai_health_analysis, daemon=True).start()

    def _run_ai_health_analysis(self):
        try:
            api_key = self.deepseek_api_key.get().strip()
            model_name = self.deepseek_model_name.get().strip() or "deepseek-chat"

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "你是一个谨慎、清晰的健康数据分析助手。只基于用户提供的数据做一般性健康建议。"},
                    {"role": "user", "content": self.build_ai_health_prompt()},
                ],
                "temperature": 0.3,
                "stream": False,
            }

            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                "https://api.deepseek.com/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            content = result["choices"][0]["message"]["content"]
            self.root.after(0, self.set_ai_result, content)

        except urllib.error.HTTPError as e:
            try:
                err = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err = str(e)
            self.root.after(0, self.set_ai_result, f"DeepSeek API 请求失败：HTTP {e.code}\\n{err}")
        except Exception as e:
            self.root.after(0, self.set_ai_result, f"AI 分析失败：{e}")

    def create_summary_item(self, parent, title, value, color, wide=False):
        width = 240 if wide else 170
        frame = tk.Frame(parent, bg="#0b0e13", highlightthickness=1, highlightbackground="#121722", width=width, height=105)
        frame.pack_propagate(False)

        tk.Label(
            frame,
            text=title,
            fg=self.text_dim,
            bg="#0b0e13",
            font=("Microsoft YaHei", 10)
        ).pack(anchor="w", padx=14, pady=(14, 8))

        value_label = tk.Label(
            frame,
            text=value,
            fg=color,
            bg="#0b0e13",
            font=("Microsoft YaHei", 17, "bold")
        )
        value_label.pack(anchor="w", padx=14)

        return {"frame": frame, "label": value_label}

    def refresh_abnormal_page(self):
        self.abnormal_page_time.config(text=time.strftime("%H:%M:%S"))

        total = len(self.health_events)
        spo2_events = [e for e in self.health_events if e["type"].startswith("spo2")]
        hr_events = [e for e in self.health_events if e["type"].startswith("hr")]
        critical_events = [e for e in self.health_events if e["level"] == "严重"]
        latest_time = self.health_events[-1]["time_str"] if self.health_events else "无"

        self.summary_total["label"].config(text=str(total))
        self.summary_spo2["label"].config(text=str(len(spo2_events)))
        self.summary_hr["label"].config(text=str(len(hr_events)))
        self.summary_critical["label"].config(text=str(len(critical_events)))
        self.summary_latest["label"].config(text=latest_time)

        for item in self.abnormal_tree.get_children():
            self.abnormal_tree.delete(item)

        for event in reversed(self.health_events):
            self.abnormal_tree.insert(
                "",
                "end",
                values=(
                    event["time_str"],
                    event["type"],
                    f"{event['value']:.1f}",
                    event["level"],
                    event["description"]
                )
            )

        self.abnormal_count_hint.config(text=f"当前 {total} 条记录")

    def clear_health_events(self):
        if not messagebox.askyesno("确认", "确定要清空所有异常记录吗？"):
            return

        self.health_events.clear()
        for k in self.last_abnormal_record_time:
            self.last_abnormal_record_time[k] = 0

        self.refresh_abnormal_page()
        self.log_message("异常记录已清空。", "info")

    def update_top_time(self):
        now = time.strftime("%H:%M:%S")
        if hasattr(self, "top_time_label"):
            self.top_time_label.config(text=now)
        if hasattr(self, "abnormal_page_time"):
            self.abnormal_page_time.config(text=now)
        self.root.after(1000, self.update_top_time)

    def log_message(self, message, tag="recv"):
        """
        主界面不再显示运行日志，避免实时接收数据时日志控件影响性能。
        日志统一写入本地 wristband_runtime.log，同时把最后一条状态显示在顶部状态条。
        """
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{now}] {message}\n"

            if self.log_file_path:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(line)

            if hasattr(self, "latest_status_text"):
                short_msg = str(message)
                if len(short_msg) > 120:
                    short_msg = short_msg[:120] + "..."
                self.latest_status_text.set(short_msg)
        except Exception:
            pass

    def clear_log(self):
        try:
            if self.log_file_path:
                with open(self.log_file_path, "w", encoding="utf-8") as f:
                    f.write("")
            self.latest_status_text.set("日志文件已清空。")
        except Exception:
            pass

    def open_log_file(self):
        try:
            if self.log_file_path and not self.log_file_path.exists():
                self.log_file_path.write_text("", encoding="utf-8")
            if sys.platform.startswith("win"):
                os.startfile(str(self.log_file_path))
            else:
                messagebox.showinfo("日志文件路径", str(self.log_file_path))
        except Exception as e:
            messagebox.showerror("无法打开日志", str(e))

    def round_rect(self, canvas, x1, y1, x2, y2, r=25, **kwargs):
        points = [
            x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r,
            x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2,
            x1+r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y2-r, x1, y1+r,
            x1, y1+r, x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def get_status_text(self, spo2, hr):
        if spo2 <= 0 or hr <= 0:
            return "等待数据", self.accent_time
        if spo2 < 90 or hr < 50 or hr > 120:
            return "需要关注", self.accent_red
        if spo2 < 95 or hr < 60 or hr > 100:
            return "轻度异常", self.warning
        return "状态良好", self.accent_green

    def get_spo2_level(self, spo2):
        if spo2 <= 0:
            return "无数据", self.text_dim
        if spo2 < 90:
            return "偏低", self.accent_red
        if spo2 < 95:
            return "略低", self.warning
        return "正常", self.accent_green

    def get_hr_level(self, hr):
        if hr <= 0:
            return "无数据", self.text_dim
        if hr < 60:
            return "偏慢", self.warning
        if hr > 100:
            return "偏快", self.accent_red
        return "正常", self.accent_green

    def draw_progress_bar(self, canvas, x1, y1, x2, y2, value, min_val, max_val, color, show_value_text="", radius=12, value_font=None):
        self.round_rect(canvas, x1, y1, x2, y2, r=radius, fill="#10141b", outline="")
        total = max(1e-6, (max_val - min_val))
        ratio = (value - min_val) / total
        ratio = max(0.0, min(1.0, ratio))
        fill_w = (x2 - x1) * ratio
        if fill_w > 4:
            self.round_rect(canvas, x1, y1, x1 + fill_w, y2, r=radius, fill=color, outline="")

        if show_value_text:
            canvas.create_text(
                x2, y1 - 10,
                text=show_value_text,
                anchor="e",
                fill=self.text_main,
                font=value_font or ("Consolas", 10, "bold")
            )

    def draw_metric_card(self, canvas, x1, y1, x2, y2, title, value_text, sub_text, value_color, scale=1.0):
        self.round_rect(canvas, x1, y1, x2, y2, r=max(12, int(18 * scale)), fill="#0b0e13", outline="#121722", width=1)
        canvas.create_text(
            x1 + 16 * scale, y1 + 18 * scale,
            text=title,
            anchor="w",
            fill=self.text_dim,
            font=("Microsoft YaHei", max(9, int(10 * scale)))
        )
        canvas.create_text(
            (x1 + x2) / 2, y1 + (y2 - y1) * 0.52,
            text=value_text,
            fill=value_color,
            font=("Segoe UI", max(18, int(24 * scale)), "bold")
        )
        canvas.create_text(
            (x1 + x2) / 2, y2 - 18 * scale,
            text=sub_text,
            fill=self.text_main,
            font=("Microsoft YaHei", max(8, int(10 * scale)), "bold")
        )

    def draw_axis_bar(self, canvas, x1, y1, x2, y2, value, color, axis_name, scale=1.0):
        self.round_rect(canvas, x1, y1, x2, y2, r=max(10, int(16 * scale)), fill="#0b0e13", outline="#121722", width=1)

        cx = (x1 + x2) / 2
        canvas.create_text(
            x1 + 14 * scale, y1 + 16 * scale,
            text=axis_name,
            anchor="w",
            fill=self.text_dim,
            font=("Microsoft YaHei", max(8, int(10 * scale)), "bold")
        )
        canvas.create_text(
            cx, y1 + 16 * scale,
            text=f"{value:.2f}",
            fill=color,
            font=("Consolas", max(9, int(11 * scale)), "bold")
        )

        line_top = y1 + 32 * scale
        line_bottom = y2 - 18 * scale
        canvas.create_line(cx, line_top, cx, line_bottom, fill="#2a3040", width=2)
        canvas.create_text(x1 + 16 * scale, y2 - 16 * scale, text="-", fill=self.text_dim, font=("Consolas", max(10, int(12 * scale)), "bold"))
        canvas.create_text(x2 - 16 * scale, y2 - 16 * scale, text="+", fill=self.text_dim, font=("Consolas", max(10, int(12 * scale)), "bold"))

        max_abs = 2.5
        v = max(-max_abs, min(max_abs, value))
        half_w = (x2 - x1 - 50 * scale) / 2
        center_y = (y1 + y2) / 2 + 10 * scale
        bar_h = max(10, 16 * scale)

        if v >= 0:
            ratio = v / max_abs
            w = half_w * ratio
            if w > 1:
                self.round_rect(
                    canvas, cx, center_y - bar_h / 2, cx + w, center_y + bar_h / 2,
                    r=max(6, int(10 * scale)), fill=color, outline=""
                )
        else:
            ratio = abs(v) / max_abs
            w = half_w * ratio
            if w > 1:
                self.round_rect(
                    canvas, cx - w, center_y - bar_h / 2, cx, center_y + bar_h / 2,
                    r=max(6, int(10 * scale)), fill=color, outline=""
                )

    def redraw_watch_panel(self):
        """
        新版仪表盘使用 Label + 小进度条，不再整块 Canvas 重绘，减少运动计步时闪烁。
        该函数名保留，兼容原有功能调用。
        """
        try:
            now_perf = time.perf_counter()
            if now_perf - self.last_dashboard_update_time < 0.08:
                if not self.dashboard_update_pending:
                    self.dashboard_update_pending = True
                    self.root.after(80, self._flush_dashboard_update)
                return

            self.last_dashboard_update_time = now_perf
            self._update_dashboard_widgets()
        except Exception:
            pass

    def _flush_dashboard_update(self):
        self.dashboard_update_pending = False
        self.last_dashboard_update_time = time.perf_counter()
        self._update_dashboard_widgets()

    def _update_dashboard_widgets(self):
        status_text, status_color = self.get_status_text(self.current_spo2, self.current_hr)

        if hasattr(self, "status_dot"):
            self.status_dot.config(fg=status_color)
        if hasattr(self, "status_text_label"):
            self.status_text_label.config(text=f"综合状态：{status_text}", fg=status_color)

        spo2_text = f"{int(self.current_spo2) if self.current_spo2 > 0 else 0}%"
        spo2_state, spo2_color = self.get_spo2_level(self.current_spo2)
        hr_text = f"{int(self.current_hr) if self.current_hr > 0 else 0}"
        hr_state, hr_color = self.get_hr_level(self.current_hr)

        self.spo2_value_label.config(text=spo2_text, fg=spo2_color if self.current_spo2 > 0 else self.accent_green)
        self.spo2_sub_label.config(text=spo2_state)
        self.set_metric_progress(self.spo2_progress, self.current_spo2, 80, 100)

        self.hr_value_label.config(text=hr_text, fg=hr_color if self.current_hr > 0 else self.accent_red)
        self.hr_sub_label.config(text=hr_state)
        self.set_metric_progress(self.hr_progress, self.current_hr, 40, 140)

        self.step_value_label.config(text=str(self.step_count))
        self.step_sub_label.config(text="今日累计")
        self.set_metric_progress(self.step_progress, min(self.step_count, 10000), 0, 10000)

        if hasattr(self, "position_value_label"):
            self.position_value_label.config(text=self.position_text)

        if hasattr(self, "imu_value_label"):
            self.imu_value_label.config(
                text=(
                    f"AX={self.x_pos:.0f}  AY={self.x_neg:.0f}  AZ={self.y_pos:.0f}  "
                    f"GX={self.y_neg:.0f}  GY={self.z_pos:.0f}  GZ={self.z_neg:.0f}"
                )
            )

    def set_bt_ui_state(self, state):
        if state == "connected":
            self.is_connected = True
            self.device_connected = True
            self.conn_btn.config(text="断开", state="normal")
            self.refresh_btn.config(state="disabled")
            self.btn_reset_steps.config(state=tk.NORMAL)
            if hasattr(self, "send_time_btn"):
                self.send_time_btn.config(state=tk.NORMAL)
            self.wristband_status_label.config(text="手环状态：已连接", fg=self.accent_green)
            self.show_toast("蓝牙连接成功", "设备已通过 SPP 连接。", kind="success")
        else:
            self.is_connected = False
            self.conn_btn.config(text="连接选中设备")
            self.refresh_btn.config(state="normal")
            self.on_device_select(None)
            self.device_connected = False
            self.wristband_status_label.config(text="手环状态：未连接", fg=self.accent_red)
            self.btn_reset_steps.config(state=tk.DISABLED)
            if hasattr(self, "send_time_btn"):
                self.send_time_btn.config(state=tk.DISABLED)

        self.redraw_watch_panel()

    def apply_responsive_layout(self, force=False):
        try:
            width = self.root.winfo_width()
            if width <= 1:
                return

            layout_mode = "horizontal" if width >= 1380 else "vertical"

            if not force and layout_mode == self.current_layout_mode:
                return

            self.current_layout_mode = layout_mode

            self.health_col.pack_forget()
            self.control_col.pack_forget()

            if layout_mode == "horizontal":
                self.health_col.pack(side="left", fill="both", expand=True, padx=(0, 6))
                self.control_col.pack(side="left", fill="both", expand=True, padx=(6, 0))
            else:
                self.health_col.pack(side="top", fill="both", expand=True, pady=(0, 8))
                self.control_col.pack(side="top", fill="both", expand=True)

            self.redraw_watch_panel()
        except Exception:
            pass

    def on_root_resize(self, event):
        if event.widget != self.root:
            return
        self.apply_responsive_layout()

    def start_ble_event_loop(self):
        if self.ble_thread is None or not self.ble_thread.is_alive():
            self.ble_thread = threading.Thread(target=self._run_ble_loop, daemon=True)
            self.ble_thread.start()
            self.log_message("BLE事件循环线程已启动。", "info")

    def _run_ble_loop(self):
        self.ble_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ble_loop)
        self.ble_loop.run_forever()

    def _schedule_ble_coro(self, coro):
        if not self.ble_loop or not self.ble_loop.is_running():
            self.log_message("BLE事件循环未运行。", "error")
            return
        asyncio.run_coroutine_threadsafe(coro, self.ble_loop)

    def on_device_select(self, event):
        if self.device_tree.selection() and not self.is_connected:
            self.conn_btn.config(state="normal")
        else:
            self.conn_btn.config(state="disabled")

    def on_protocol_change(self, event=None):
        # 当前版本界面只开放 SPP，BLE 相关代码保留但不作为主界面功能入口。
        self.bt_type.set("SPP")
        for item in self.device_tree.get_children():
            self.device_tree.delete(item)
        self.selected_device_addr = None
        self.conn_btn.config(state="disabled")
        self.log_message("连接类型固定为：SPP 串口透传", "info")

    def on_device_double_click(self, event):
        if self.device_tree.selection():
            self.toggle_connection()

    def scan_devices(self):
        self.bt_type.set("SPP")
        self.log_message("开始扫描 SPP 设备...", "info")
        self.refresh_btn.config(state="disabled")

        for item in self.device_tree.get_children():
            self.device_tree.delete(item)

        threading.Thread(target=self._scan_spp, daemon=True).start()

    def _scan_spp(self):
        try:
            devices = bluetooth.discover_devices(duration=8, lookup_names=True, flush_cache=True)
            self.root.after(0, self._on_scan_complete, devices)
        except Exception as e:
            self.root.after(0, self.log_message, f"SPP扫描失败: {e}", "error")
            self.root.after(0, self._on_scan_complete, None)

    async def _scan_ble(self):
        try:
            devices = await BleakScanner.discover()
            result = [(d.address, d.name) for d in devices]
            self.root.after(0, self._on_scan_complete, result)
        except Exception as e:
            self.root.after(0, self.log_message, f"BLE扫描失败: {e}", "error")
            self.root.after(0, self._on_scan_complete, None)

    def _on_scan_complete(self, devices):
        self.refresh_btn.config(state="normal")

        if not devices:
            self.log_message("未发现任何蓝牙设备。", "info")
            self.conn_btn.config(state="disabled")
            return

        self.log_message(f"扫描完成，共发现 {len(devices)} 个设备：", "success")
        for addr, name in devices:
            show_name = name if name else "Unknown"
            self.device_tree.insert("", "end", values=(addr, show_name))

        self.on_device_select(None)

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect_bt()
        else:
            self.connect_bt()

    def connect_bt(self):
        selection = self.device_tree.selection()
        if not selection:
            messagebox.showwarning("未选择设备", "请先选择一个蓝牙设备")
            return

        item = self.device_tree.item(selection[0])
        self.selected_device_addr = item["values"][0]

        self.bt_type.set("SPP")
        self.conn_btn.config(state="disabled")
        self.log_message(f"开始连接设备：{self.selected_device_addr}", "info")

        threading.Thread(target=self._connect_spp, args=(self.selected_device_addr,), daemon=True).start()

    def _connect_spp(self, addr):
        try:
            self.bt_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.bt_socket.connect((addr, self.bt_port.get()))
            self.root.after(0, self.log_message, f"[SPP] 成功连接到 {addr}:{self.bt_port.get()}", "success")
            self.root.after(0, self.set_bt_ui_state, "connected")
            self.stop_event.clear()
            threading.Thread(target=self._spp_recv_loop, daemon=True).start()
        except Exception as e:
            self.root.after(0, self.log_message, f"[SPP] 连接失败: {e}", "error")
            self.root.after(0, self.set_bt_ui_state, "disconnected")

    async def _connect_ble(self, addr):
        try:
            self.ble_client = BleakClient(addr)
            await self.ble_client.connect()
            self.root.after(0, self.log_message, f"[BLE] 成功连接到 {addr}", "success")

            await self.ble_client.start_notify(
                self.ble_char_notify_uuid.get(),
                lambda sender, data: self.root.after(0, self.on_ble_data_received, sender, data),
            )

            self.root.after(0, self.log_message, "[BLE] 已启用通知。", "success")
            self.root.after(0, self.set_bt_ui_state, "connected")
        except Exception as e:
            self.root.after(0, self.log_message, f"[BLE] 连接失败: {e}", "error")
            self.root.after(0, self.set_bt_ui_state, "disconnected")


    def send_pc_time_to_device(self):
        """
        向手环发送电脑当前时间。

        发送格式与手机软件保持一致：
        TIME:HHmmss\n
        示例：
        TIME:232145\n
        当前电脑端主界面固定使用 SPP 串口透传；BLE 发送逻辑保留，避免以后恢复 BLE 时无法使用。
        """
        if not self.is_connected:
            messagebox.showwarning("未连接", "请先连接手环后再发送电脑时间。")
            return

        time_text = time.strftime("%H%M%S")
        command = f"TIME:{time_text}\n"
        data = command.encode("utf-8")

        if self.bt_type.get() == "SPP":
            if not self.bt_socket:
                messagebox.showwarning("未连接", "SPP 蓝牙连接不存在，请重新连接手环。")
                return

            try:
                self.bt_socket.send(data)
                self.log_message(f"[SPP发送] {command.strip()}", "success")
                self.show_toast("发送成功", f"已发送电脑时间：{time_text}", kind="success")
            except Exception as e:
                self.log_message(f"[SPP发送失败] {e}", "error")
                messagebox.showerror("发送失败", f"电脑时间发送失败：{e}")

            return

        if self.bt_type.get() == "BLE":
            if not self.ble_client:
                messagebox.showwarning("未连接", "BLE 蓝牙连接不存在，请重新连接手环。")
                return

            self._schedule_ble_coro(self._send_pc_time_to_device_ble(data, time_text))
            return

        messagebox.showwarning("发送失败", "未知蓝牙连接类型。")

    async def _send_pc_time_to_device_ble(self, data, time_text):
        """BLE 发送电脑当前时间，格式：TIME:HHmmss\n。"""
        try:
            if not self.ble_client or not self.ble_client.is_connected:
                self.root.after(0, messagebox.showwarning, "未连接", "BLE 尚未连接。")
                return

            await self.ble_client.write_gatt_char(
                self.ble_char_write_uuid.get(),
                data,
                response=False
            )

            self.root.after(0, self.log_message, f"[BLE发送] TIME:{time_text}", "success")
            self.root.after(0, self.show_toast, "发送成功", f"已发送电脑时间：{time_text}", "success")

        except Exception as e:
            self.root.after(0, self.log_message, f"[BLE发送失败] {e}", "error")
            self.root.after(0, messagebox.showerror, "发送失败", f"电脑时间发送失败：{e}")

    def disconnect_bt(self):
        self.stop_event.set()
        self.log_message("开始断开蓝牙连接。", "info")

        if self.bt_type.get() == "SPP" and self.bt_socket:
            try:
                self.bt_socket.close()
            except Exception:
                pass
            self.bt_socket = None

        elif self.bt_type.get() == "BLE" and self.ble_client:
            self._schedule_ble_coro(self._disconnect_ble())

        self.set_bt_ui_state("disconnected")
        self.log_message("已手动断开蓝牙连接。", "info")

    async def _disconnect_ble(self):
        if self.ble_client and self.ble_client.is_connected:
            try:
                await self.ble_client.stop_notify(self.ble_char_notify_uuid.get())
                await self.ble_client.disconnect()
            except Exception:
                pass
        self.ble_client = None

    def _spp_recv_loop(self):
        while not self.stop_event.is_set() and self.bt_socket:
            try:
                data = self.bt_socket.recv(1024)
                if not data:
                    break
                text_str = data.decode("utf-8", errors="backslashreplace")
                self.root.after(0, self.log_message, f"[SPP接收] {text_str}", "recv")
                self.root.after(0, self.process_received_text, text_str)
            except Exception:
                break

        self.root.after(0, self.set_bt_ui_state, "disconnected")

    def on_ble_data_received(self, sender, data):
        try:
            text_str = data.decode("utf-8", errors="backslashreplace")
            self.log_message(f"[BLE接收] {text_str}", "recv")
            self.process_received_text(text_str)
        except Exception as e:
            self.log_message(f"[BLE] 数据处理异常: {e}", "error")

    def refresh_axis_data(self):
        self.current_x = self.x_pos - self.x_neg
        self.current_y = self.y_pos - self.y_neg
        self.current_z = self.z_pos - self.z_neg

    def update_position_text(self):
        def axis_desc(v, axis_name):
            if v > 0.05:
                return f"{axis_name}+"
            elif v < -0.05:
                return f"{axis_name}-"
            return f"{axis_name}0"

        self.position_text = f"{axis_desc(self.current_x, 'X')} / {axis_desc(self.current_y, 'Y')} / {axis_desc(self.current_z, 'Z')}"

    def _append_health_history(self):
        try:
            now_ts = time.time()
            self.history_time.append(now_ts)
            self.hr_history.append(float(self.current_hr))
            self.spo2_history.append(float(self.current_spo2))
        except Exception:
            pass

    def show_toast(self, title, message, kind="info", duration_ms=3500):
        try:
            popup = tk.Toplevel(self.root)
            popup.title(title)
            popup.configure(bg=self.card)
            popup.resizable(False, False)
            popup.attributes("-topmost", True)

            color = self.accent_green if kind == "success" else self.accent_red if kind == "error" else self.accent_time
            popup.geometry("+{}+{}".format(self.root.winfo_rootx() + self.root.winfo_width() - 380, self.root.winfo_rooty() + 80))

            tk.Label(
                popup,
                text=title,
                bg=self.card,
                fg=color,
                font=("Microsoft YaHei", 13, "bold"),
            ).pack(anchor="w", padx=18, pady=(16, 6))

            tk.Label(
                popup,
                text=message,
                bg=self.card,
                fg=self.text_main,
                font=("Microsoft YaHei", 10),
                wraplength=320,
                justify="left",
            ).pack(anchor="w", padx=18, pady=(0, 14))

            btn = self.dark_button(popup, "关闭", popup.destroy, width=8)
            btn.pack(anchor="e", padx=18, pady=(0, 16))

            popup.after(duration_ms, lambda: popup.winfo_exists() and popup.destroy())
        except Exception:
            pass

    def show_health_alert_popup(self, event_type, value, level, description):
        try:
            popup = tk.Toplevel(self.root)
            popup.title("健康异常提醒")
            popup.configure(bg="#3a0710")
            popup.resizable(False, False)
            popup.attributes("-topmost", True)
            popup.geometry("+{}+{}".format(self.root.winfo_rootx() + self.root.winfo_width() - 420, self.root.winfo_rooty() + 150))

            title = tk.Label(
                popup,
                text="健康异常提醒",
                bg="#3a0710",
                fg="#ffffff",
                font=("Microsoft YaHei", 15, "bold"),
            )
            title.pack(anchor="w", padx=20, pady=(18, 8))

            info = tk.Label(
                popup,
                text=f"{event_type}：{value:.1f}\\n等级：{level}\\n{description}",
                bg="#3a0710",
                fg="#ffe4e6",
                font=("Microsoft YaHei", 11, "bold"),
                justify="left",
                wraplength=360,
            )
            info.pack(anchor="w", padx=20, pady=(0, 14))

            btn = self.dark_button(popup, "我知道了", popup.destroy, width=10)
            btn.pack(anchor="e", padx=20, pady=(0, 18))

            def flash(count=0):
                try:
                    if not popup.winfo_exists():
                        return
                    popup.configure(bg="#7f1d1d" if count % 2 == 0 else "#3a0710")
                    title.configure(bg=popup.cget("bg"))
                    info.configure(bg=popup.cget("bg"))
                    if count < 8:
                        popup.after(260, lambda: flash(count + 1))
                except Exception:
                    pass

            flash()
        except Exception:
            pass

    def show_trend_chart_popup(self):
        try:
            win = tk.Toplevel(self.root)
            win.title("心率 / 血氧趋势图")
            win.geometry("860x520")
            win.configure(bg=self.bg)

            header = tk.Frame(win, bg=self.bg)
            header.pack(fill="x", padx=16, pady=(14, 8))

            tk.Label(
                header,
                text="心率 / 血氧趋势图",
                bg=self.bg,
                fg=self.text_main,
                font=("Microsoft YaHei", 16, "bold"),
            ).pack(side="left")

            self.dark_button(header, "关闭", win.destroy, width=8).pack(side="right")

            canvas = tk.Canvas(win, bg=self.card, highlightthickness=1, highlightbackground=self.border)
            canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

            def draw_chart(event=None):
                canvas.delete("all")
                w = max(300, canvas.winfo_width())
                h = max(220, canvas.winfo_height())
                pad_l, pad_r, pad_t, pad_b = 60, 30, 35, 45
                x1, y1, x2, y2 = pad_l, pad_t, w - pad_r, h - pad_b

                canvas.create_rectangle(x1, y1, x2, y2, outline=self.border, fill="#0b1320")
                canvas.create_text(x1, 16, text="最近数据趋势", anchor="w", fill=self.text_main, font=("Microsoft YaHei", 11, "bold"))

                hr_vals = list(self.hr_history)[-120:]
                spo2_vals = list(self.spo2_history)[-120:]

                if len(hr_vals) < 2 and len(spo2_vals) < 2:
                    canvas.create_text(w / 2, h / 2, text="暂无足够数据", fill=self.text_dim, font=("Microsoft YaHei", 14, "bold"))
                    return

                for i in range(5):
                    yy = y1 + (y2 - y1) * i / 4
                    canvas.create_line(x1, yy, x2, yy, fill="#1f2d44")
                    canvas.create_text(x1 - 8, yy, text=str(100 - i * 25), anchor="e", fill=self.text_dim, font=("Consolas", 9))

                def draw_series(vals, min_v, max_v, color, name, y_offset=0):
                    if len(vals) < 2:
                        return
                    pts = []
                    n = len(vals)
                    for idx, val in enumerate(vals):
                        xx = x1 + (x2 - x1) * idx / max(1, n - 1)
                        ratio = (val - min_v) / max(1e-6, (max_v - min_v))
                        ratio = max(0, min(1, ratio))
                        yy = y2 - ratio * (y2 - y1)
                        pts.extend([xx, yy])
                    canvas.create_line(*pts, fill=color, width=2, smooth=True)
                    canvas.create_text(x2 - 120, y1 + 18 + y_offset, text=name, anchor="w", fill=color, font=("Microsoft YaHei", 10, "bold"))

                draw_series(spo2_vals, 80, 100, self.accent_green, "SpO₂ 80-100%", 0)
                draw_series(hr_vals, 40, 140, self.accent_red, "心率 40-140", 22)

            canvas.bind("<Configure>", draw_chart)
            win.after(200, draw_chart)
        except Exception as e:
            messagebox.showerror("趋势图错误", str(e))

    def record_health_abnormality(self, event_type, value, level, description):
        now_ts = time.time()
        last_ts = self.last_abnormal_record_time.get(event_type, 0)

        if now_ts - last_ts < self.abnormal_record_cooldown:
            return

        self.last_abnormal_record_time[event_type] = now_ts
        record = {
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts)),
            "timestamp": now_ts,
            "type": event_type,
            "value": value,
            "level": level,
            "description": description
        }
        self.health_events.append(record)
        self.log_message(
            f"[健康异常记录] 时间：{record['time_str']} | 类型：{event_type} | 数值：{value:.1f} | 等级：{level} | {description} 🩺",
            "warning"
        )
        self.show_health_alert_popup(event_type, value, level, description)


        if hasattr(self, "abnormal_tree") and self.abnormal_page.winfo_ismapped():
            self.refresh_abnormal_page()

    def check_health_abnormalities(self):
        if self.current_spo2 > 0:
            if self.current_spo2 < 90:
                self.record_health_abnormality(
                    "spo2_critical",
                    self.current_spo2,
                    "严重",
                    "血氧低于90%，存在明显低氧风险"
                )
            elif self.current_spo2 < 95:
                self.record_health_abnormality(
                    "spo2_low",
                    self.current_spo2,
                    "轻度",
                    "血氧低于95%，建议休息并观察"
                )

        if self.current_hr > 0:
            if self.current_hr < 50:
                self.record_health_abnormality(
                    "hr_critical_low",
                    self.current_hr,
                    "严重",
                    "心率低于50次/分，建议重点关注"
                )
            elif self.current_hr < 60:
                self.record_health_abnormality(
                    "hr_low",
                    self.current_hr,
                    "轻度",
                    "心率低于60次/分，可能偏慢"
                )
            elif self.current_hr > 120:
                self.record_health_abnormality(
                    "hr_critical_high",
                    self.current_hr,
                    "严重",
                    "心率高于120次/分，建议尽快休息并关注"
                )
            elif self.current_hr > 100:
                self.record_health_abnormality(
                    "hr_high",
                    self.current_hr,
                    "轻度",
                    "心率高于100次/分，可能偏快"
                )

    def summarize_health_events(self):
        if not self.health_events:
            return "最近没有记录到明显的心率/血氧异常。"

        total = len(self.health_events)
        spo2_events = [e for e in self.health_events if e["type"].startswith("spo2")]
        hr_events = [e for e in self.health_events if e["type"].startswith("hr")]
        critical_events = [e for e in self.health_events if e["level"] == "严重"]

        latest = self.health_events[-1]
        latest_text = (
            f"最近一次异常：{latest['time_str']}，"
            f"{latest['type']}，数值 {latest['value']:.1f}，"
            f"等级 {latest['level']}。"
        )

        summary = []
        summary.append(f"共记录异常事件 {total} 次。")
        summary.append(f"其中血氧异常 {len(spo2_events)} 次，心率异常 {len(hr_events)} 次。")
        summary.append(f"严重异常 {len(critical_events)} 次。")
        summary.append(latest_text)

        if len(critical_events) >= 3:
            summary.append("近期严重异常次数较多，建议尽快进行人工复查或医学评估。")
        elif len(spo2_events) >= 3 and len(hr_events) >= 3:
            summary.append("血氧和心率均多次出现异常，说明健康状态波动较明显。")
        elif len(spo2_events) >= 3:
            summary.append("血氧异常出现较频繁，建议关注呼吸状态、佩戴质量及运动负荷。")
        elif len(hr_events) >= 3:
            summary.append("心率异常出现较频繁，建议关注情绪、活动强度及休息情况。")
        else:
            summary.append("异常事件数量暂不多，但仍建议持续观察变化趋势。")

        return "\n".join(summary)

    def get_recent_health_events_text(self, max_count=10):
        if not self.health_events:
            return "无异常记录。"
        recent = list(self.health_events)[-max_count:]
        lines = []
        for e in recent:
            lines.append(
                f"{e['time_str']} | {e['type']} | {e['value']:.1f} | {e['level']} | {e['description']}"
            )
        return "\n".join(lines)

    def process_received_text(self, text):
        self.rx_text_buffer += text
        normalized = self.rx_text_buffer.replace("\r", "\n")

        def find_last_value(prefix):
            """
            同时兼容两种格式：
            1. 老格式：
               a:78 b:98 c:120 d:-35 e:980 f:5 g:-3 h:8
            2. 你的硬件当前格式：
               a78b98c120d-35e980f5g-3h8
            """
            pattern = rf"{prefix}\s*:?\s*(-?\d+(?:\.\d+)?)(?=[a-hA-H]|$|\s|\r|\n)"
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])
                except Exception:
                    return None
            return None

        updated = False
        axis_updated = False
        health_value_updated = False

        hr_val = find_last_value("a")
        if hr_val is not None:
            self.current_hr = hr_val
            updated = True
            health_value_updated = True

        spo2_val = find_last_value("b")
        if spo2_val is not None:
            self.current_spo2 = spo2_val
            updated = True
            health_value_updated = True

        c_val = find_last_value("c")
        d_val = find_last_value("d")
        e_val = find_last_value("e")
        f_val = find_last_value("f")
        g_val = find_last_value("g")
        h_val = find_last_value("h")
        t_val = find_last_value("t")  # 可选：硬件毫秒时间戳，例如 t123456a0b0c...

        if c_val is not None:
            self.x_pos = c_val
            axis_updated = True
        if d_val is not None:
            self.x_neg = d_val
            axis_updated = True
        if e_val is not None:
            self.y_pos = e_val
            axis_updated = True
        if f_val is not None:
            self.y_neg = f_val
            axis_updated = True
        if g_val is not None:
            self.z_pos = g_val
            axis_updated = True
        if h_val is not None:
            self.z_neg = h_val
            axis_updated = True

        if health_value_updated:
            self._append_health_history()
            self.check_health_abnormalities()

        if axis_updated:
            self.refresh_axis_data()
            # 模型计步需要 6 轴输入。这里约定：c,d,e,f,g,h 分别为 ax,ay,az,gx,gy,gz。
            # 如果你的设备发送的不是这个顺序，需要先在这里改映射。
            if all(v is not None for v in [c_val, d_val, e_val, f_val, g_val, h_val]):
                # 有 t 字段时优先使用硬件毫秒时间戳；没有 t 字段时使用电脑接收时间。
                # 硬件时间戳格式示例：t123456a0b0c178d43e2295f-78g62h30
                timestamp_s = (t_val / 1000.0) if t_val is not None else None
                self.process_raw_imu_frame(c_val, d_val, e_val, f_val, g_val, h_val, timestamp_s=timestamp_s)
            self.update_position_text()
            updated = True

        if updated:
            self.device_connected = True
            self.wristband_status_label.config(text="手环状态：数据正常", fg=self.accent_green)
            self.redraw_watch_panel()

        if len(self.rx_text_buffer) > 800:
            self.rx_text_buffer = self.rx_text_buffer[-800:]

    def reset_steps(self):
        self.step_count = 0
        self.imu_buffer.clear()
        self.model_prob_queue.clear()
        self.raw_imu_time_buffer.clear()
        self.input_time_buffer.clear()
        self.next_resample_time = None
        self.imu_frame_index = 0
        self.frames_since_infer = 0
        self.last_model_step_frame = -999999
        self.redraw_watch_panel()
        self.log_message("步数已重置为 0，模型计步缓冲区已清空。", "info")

    def calculate_magnitude(self, x, y, z):
        return math.sqrt(x * x + y * y + z * z)

    def _get_step_engine_error(self):
        """读取 C++ DLL 返回的错误信息。"""
        try:
            if self.step_engine_dll is None:
                return "step_engine.dll 未加载"
            raw = self.step_engine_dll.get_last_error()
            if not raw:
                return "无错误信息"
            return raw.decode("utf-8", errors="ignore")
        except Exception as e:
            return f"读取 DLL 错误信息失败：{e}"

    def init_step_model(self):
        """加载 StepPeakNet-25Hz 的 C++ DLL、ONNX 模型和标准化参数。"""
        if np is None:
            self.step_model_enabled = False
            self.log_message("模型计步未启用：缺少 numpy。", "warning")
            return

        onnx_path = self.model_paths["onnx"]
        norm_path = self.model_paths["norm"]
        dll_dir = self.model_paths["dll_dir"]
        dll_path = self.model_paths["dll"]

        if not dll_path.exists():
            self.step_model_enabled = False
            self.log_message(f"DLL模型计步未启用：找不到 DLL 文件 {dll_path}", "warning")
            self.log_message("请确认 step_engine_runtime\\step_engine.dll 是否存在。", "warning")
            return

        if not onnx_path.exists():
            self.step_model_enabled = False
            self.log_message(f"DLL模型计步未启用：找不到 ONNX 模型 {onnx_path}", "warning")
            self.log_message("请确认 model_run_v2\\steppeaknet_25hz.onnx 是否存在。", "warning")
            return

        if not norm_path.exists():
            self.step_model_enabled = False
            self.log_message(f"DLL模型计步未启用：找不到标准化参数 {norm_path}", "warning")
            self.log_message("请确认 processed_dataset\\norm_stats.json 是否存在。", "warning")
            return

        try:
            with open(norm_path, "r", encoding="utf-8") as f:
                norm = json.load(f)
            self.step_norm_mean = np.array(norm["mean"], dtype=np.float32)
            self.step_norm_std = np.array(norm["std"], dtype=np.float32)
            self.step_norm_std[self.step_norm_std < 1e-6] = 1e-6

            # Python 3.8+ 在 Windows 加载 DLL 依赖时，需要把 DLL 所在目录加入搜索路径。
            if hasattr(os, "add_dll_directory"):
                self.step_dll_directory_cookie = os.add_dll_directory(str(dll_dir))

            self.step_engine_dll = ctypes.CDLL(str(dll_path))

            self.step_engine_dll.get_last_error.argtypes = []
            self.step_engine_dll.get_last_error.restype = ctypes.c_char_p

            self.step_engine_dll.init_model.argtypes = [ctypes.c_wchar_p]
            self.step_engine_dll.init_model.restype = ctypes.c_int

            self.step_engine_dll.run_model.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            self.step_engine_dll.run_model.restype = ctypes.c_int

            self.step_engine_dll.release_model.argtypes = []
            self.step_engine_dll.release_model.restype = None

            ret = self.step_engine_dll.init_model(str(onnx_path))
            if ret != 0:
                err = self._get_step_engine_error()
                self.step_model_enabled = False
                self.log_message(f"DLL模型计步初始化失败：init_model 返回 {ret}，{err}", "error")
                return

            self.step_model_enabled = True
            self.log_message(f"StepPeakNet-25Hz DLL模型计步已启用：{onnx_path}", "success")
            self.log_message(f"DLL目录：{dll_dir}", "info")

        except Exception as e:
            self.step_model_enabled = False
            self.log_message(f"DLL模型计步初始化失败：{e}", "error")

    def build_model_features(self, raw6):
        """把 6 轴原始数据转换成训练时一致的 8 维特征。"""
        ax, ay, az, gx, gy, gz = raw6
        acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        return [ax, ay, az, gx, gy, gz, acc_mag, gyro_mag]

    def process_raw_imu_frame(self, ax, ay, az, gx, gy, gz, timestamp_s=None):
        """
        接收蓝牙原始六轴数据。

        这里的数据频率可能不是稳定 25Hz，例如 22Hz、24Hz、28Hz。
        本函数只负责把原始数据带时间戳放入缓存，然后触发 25Hz 重采样。
        """
        if not self.step_model_enabled:
            return

        try:
            raw6 = [float(ax), float(ay), float(az), float(gx), float(gy), float(gz)]
            now = time.perf_counter() if timestamp_s is None else float(timestamp_s)

            # 硬件时间戳重启或回绕时，重置重采样状态，避免时间倒退造成缓存卡死。
            if self.raw_imu_time_buffer and now <= self.raw_imu_time_buffer[-1][0]:
                self._reset_resample_buffers("IMU时间戳倒退，重采样缓存已重置。")

            self.raw_imu_time_buffer.append((now, raw6))
            self.input_time_buffer.append(now)

            if self.resample_enabled:
                self._resample_to_25hz()
            else:
                self.process_step_model_frame_25hz(*raw6)

        except Exception as e:
            self.log_message(f"原始IMU处理异常：{e}", "error")

    def _resample_to_25hz(self):
        """把不稳定输入频率整理成固定 25Hz，再送入模型窗口。"""
        if len(self.raw_imu_time_buffer) < 2:
            return

        if self.next_resample_time is None:
            # 从第一帧原始数据时间开始，之后每 0.04 秒生成一帧虚拟 25Hz 数据。
            self.next_resample_time = self.raw_imu_time_buffer[0][0]

        generated = 0
        while len(self.raw_imu_time_buffer) >= 2:
            latest_t = self.raw_imu_time_buffer[-1][0]
            if self.next_resample_time > latest_t:
                break

            pair = self._find_surrounding_raw_frames(self.next_resample_time)
            if pair is None:
                break

            t0, v0, t1, v1 = pair
            gap = t1 - t0

            # 蓝牙卡顿太久时，不要用很长跨度的插值硬补，否则容易误计步。
            if gap <= 0 or gap > self.max_interpolate_gap:
                self._reset_resample_buffers("IMU数据间隔过大，重采样缓存已重置。")
                return

            ratio = (self.next_resample_time - t0) / gap
            raw6_interp = [v0[i] + ratio * (v1[i] - v0[i]) for i in range(6)]

            self.process_step_model_frame_25hz(*raw6_interp)
            self.next_resample_time += self.target_dt
            generated += 1

            # 防止异常情况下 while 一次生成过多帧卡住界面。
            if generated >= 20:
                break

        self._trim_old_raw_frames()
        self._log_resample_status()

    def _find_surrounding_raw_frames(self, target_t):
        """找到 target_t 前后的两帧原始数据，用于线性插值。"""
        buf = list(self.raw_imu_time_buffer)
        for i in range(len(buf) - 1):
            t0, v0 = buf[i]
            t1, v1 = buf[i + 1]
            if t0 <= target_t <= t1:
                return t0, v0, t1, v1
        return None

    def _trim_old_raw_frames(self):
        """删除已经用不到的旧原始帧，同时保留一帧用于后续插值。"""
        if self.next_resample_time is None:
            return
        while len(self.raw_imu_time_buffer) >= 3:
            if self.raw_imu_time_buffer[1][0] < self.next_resample_time:
                self.raw_imu_time_buffer.popleft()
            else:
                break

    def _reset_resample_buffers(self, message=None):
        """采样时间异常或蓝牙卡顿时，清空重采样和模型缓存。"""
        self.raw_imu_time_buffer.clear()
        self.input_time_buffer.clear()
        self.next_resample_time = None
        self.imu_buffer.clear()
        self.model_prob_queue.clear()
        self.frames_since_infer = 0

        now = time.perf_counter()
        if message and now - self.last_resample_reset_log_time > 2.0:
            self.last_resample_reset_log_time = now
            self.log_message(message, "warning")

    def _log_resample_status(self):
        """每隔几秒输出一次实际输入Hz，方便现场判断硬件频率。"""
        now = time.perf_counter()
        if now - self.last_resample_log_time < 5.0:
            return
        self.last_resample_log_time = now

        if len(self.input_time_buffer) >= 2:
            duration = self.input_time_buffer[-1] - self.input_time_buffer[0]
            if duration > 0:
                hz = (len(self.input_time_buffer) - 1) / duration
                self.log_message(f"采样率适配：输入约 {hz:.1f}Hz → 模型固定 25.0Hz", "info")

    def process_step_model_frame_25hz(self, ax, ay, az, gx, gy, gz):
        """
        固定 25Hz 模型计步入口。

        重要约定：这里接收的已经是重采样后的 25Hz 六轴数据。
        c,d,e,f,g,h 分别对应 ax,ay,az,gx,gy,gz。
        """
        if not self.step_model_enabled:
            return

        try:
            raw6 = [float(ax), float(ay), float(az), float(gx), float(gy), float(gz)]
            feat8 = self.build_model_features(raw6)
            self.imu_buffer.append(feat8)
            self.imu_frame_index += 1
            self.frames_since_infer += 1

            if len(self.imu_buffer) < self.model_window_size:
                return
            if self.frames_since_infer < self.model_stride:
                return
            self.frames_since_infer = 0

            window = np.array(self.imu_buffer, dtype=np.float32)  # [64, 8]
            window_norm = (window - self.step_norm_mean) / self.step_norm_std
            window_norm = np.ascontiguousarray(window_norm, dtype=np.float32)

            step_prob = np.zeros(64, dtype=np.float32)
            gait_prob_arr = np.zeros(1, dtype=np.float32)

            ret = self.step_engine_dll.run_model(
                window_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                step_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                gait_prob_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if ret != 0:
                err = self._get_step_engine_error()
                self.log_message(f"DLL模型推理失败：run_model 返回 {ret}，{err}", "error")
                return

            gait_prob = float(gait_prob_arr[0])

            # 为了减少窗口边缘误差，只提交窗口中间附近的 5 帧预测。
            # 这样会有约 1.3 秒延迟，但比直接用末尾帧更稳。
            commit_start = self.model_window_size // 2 - self.model_stride // 2
            commit_end = commit_start + self.model_stride
            window_start_frame = self.imu_frame_index - self.model_window_size

            for local_idx in range(commit_start, commit_end):
                global_frame = window_start_frame + local_idx
                prob = float(step_prob[local_idx])
                self._consume_model_probability(global_frame, prob, gait_prob)

        except Exception as e:
            self.log_message(f"模型计步异常：{e}", "error")

    def _consume_model_probability(self, frame_idx, step_prob, gait_prob):
        """用 3 点局部峰值 + gait 门控，把模型概率转换为步数。"""
        self.model_prob_queue.append((frame_idx, step_prob, gait_prob))
        if len(self.model_prob_queue) < 3:
            return

        prev_item, curr_item, next_item = list(self.model_prob_queue)
        curr_frame, curr_prob, curr_gait = curr_item

        is_peak = curr_prob > prev_item[1] and curr_prob >= next_item[1]
        strong_enough = curr_prob >= self.model_step_threshold
        gait_ok = curr_gait >= self.model_gait_threshold
        separated = (curr_frame - self.last_model_step_frame) >= self.model_min_distance

        if is_peak and strong_enough and gait_ok and separated:
            self.step_count += 1
            self.last_model_step_frame = curr_frame
            self.log_message(
                f"[模型计步] 当前步数：{self.step_count} | step={curr_prob:.3f} | gait={curr_gait:.3f}",
                "success"
            )
            self.redraw_watch_panel()

    def detect_step(self, x, y, z):
        """旧版阈值峰值计步已废弃。保留空函数，避免旧调用导致程序报错。"""
        return

    def on_closing(self):
        if self.is_connected:
            self.disconnect_bt()

        if self.ble_loop:
            try:
                self.ble_loop.call_soon_threadsafe(self.ble_loop.stop)
            except Exception:
                pass

        if self.ble_thread and self.ble_thread.is_alive():
            self.ble_thread.join(timeout=1)

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BluetoothWristbandAssistant(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()