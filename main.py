
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import csv
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

W98 = {
    'bg': '#c0c0c0',
    'bg_dark': '#808080',
    'bg_light': '#ffffff',
    'title_bg': '#000080',
    'title_fg': '#ffffff',
    'btn_bg': '#c0c0c0',
    'btn_active': '#c0c0c0',
    'text': '#000000',
    'disabled': '#808080',
    'highlight': '#000080',
    'console_bg': '#000000',
    'console_fg': '#c0c0c0',
    'select_bg': '#000080',
    'select_fg': '#ffffff',
    'font': ('Microsoft Sans Serif', 8),
    'font_bold': ('Microsoft Sans Serif', 8, 'bold'),
    'font_title': ('Microsoft Sans Serif', 10, 'bold'),
    'font_mono': ('Courier New', 8),
}


def w98_frame(parent, **kw):
    return tk.Frame(parent, bg=W98['bg'], **kw)


def w98_label(parent, text, bold=False, **kw):
    font = W98['font_bold'] if bold else W98['font']
    return tk.Label(parent, text=text, bg=W98['bg'], fg=W98['text'], font=font, **kw)


def w98_button(parent, text, command, width=None, **kw):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=W98['btn_bg'], fg=W98['text'], font=W98['font'],
        relief=tk.RAISED, bd=2,
        activebackground=W98['btn_active'], activeforeground=W98['text'],
        cursor='arrow',
        **({} if width is None else {'width': width}), **kw)
    return btn


def w98_entry(parent, textvariable=None, width=10, **kw):
    return tk.Entry(parent, bg=W98['bg_light'], fg=W98['text'],
                    font=W98['font'], relief=tk.SUNKEN, bd=2,
                    insertbackground=W98['text'],
                    textvariable=textvariable, width=width, **kw)


def w98_listbox(parent, **kw):
    return tk.Listbox(parent, bg=W98['bg_light'], fg=W98['text'],
                      font=W98['font_mono'],
                      selectbackground=W98['select_bg'],
                      selectforeground=W98['select_fg'],
                      relief=tk.SUNKEN, bd=2, **kw)


def w98_scrollbar(parent, **kw):
    return tk.Scrollbar(parent, **kw)


def w98_labelframe(parent, text, **kw):
    outer = tk.Frame(parent, bg=W98['bg'], **kw)
    tk.Label(outer, text=f" {text} ", bg=W98['bg'], fg=W98['text'],
             font=W98['font_bold']).pack(anchor=tk.W, padx=4)
    inner = tk.Frame(outer, bg=W98['bg'], relief=tk.GROOVE, bd=2)
    inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    return outer, inner


def w98_title_bar(parent, title, icon="🖥"):
    bar = tk.Frame(parent, bg=W98['title_bg'], height=20)
    bar.pack(fill=tk.X)
    bar.pack_propagate(False)
    tk.Label(bar, text=f"  {icon}  {title}", bg=W98['title_bg'], fg=W98['title_fg'],
             font=W98['font_bold'], anchor=tk.W).pack(side=tk.LEFT, fill=tk.Y)
    return bar


def w98_separator(parent):
    tk.Frame(parent, bg=W98['bg_dark'], height=1).pack(fill=tk.X, pady=2)
    tk.Frame(parent, bg='white', height=1).pack(fill=tk.X)


class W98Notebook:
    def __init__(self, parent):
        self.parent = parent
        self.tabs = []
        self.current = 0

        self.tab_bar = tk.Frame(parent, bg=W98['bg'])
        self.tab_bar.pack(fill=tk.X)

        self.content_border = tk.Frame(parent, bg=W98['bg_dark'], bd=1, relief=tk.RAISED)
        self.content_border.pack(fill=tk.BOTH, expand=True)

        self.content = tk.Frame(self.content_border, bg=W98['bg'])
        self.content.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    def add(self, text):
        idx = len(self.tabs)
        frame = tk.Frame(self.content, bg=W98['bg'])
        btn = tk.Button(
            self.tab_bar, text=text,
            font=W98['font'],
            bg=W98['bg'] if idx == 0 else W98['bg_dark'],
            fg=W98['text'],
            relief=tk.RAISED if idx == 0 else tk.FLAT,
            bd=2, padx=6, pady=2,
            command=lambda i=idx: self.select(i))
        btn.pack(side=tk.LEFT, padx=(2 if idx == 0 else 0, 0), pady=(4, 0))
        self.tabs.append((text, frame, btn))
        if idx == 0:
            frame.pack(fill=tk.BOTH, expand=True)
        return frame

    def select(self, idx):
        if 0 <= idx < len(self.tabs):
            _, old_frame, old_btn = self.tabs[self.current]
            old_frame.pack_forget()
            old_btn.config(bg=W98['bg_dark'], relief=tk.FLAT)
            self.current = idx
            _, new_frame, new_btn = self.tabs[idx]
            new_frame.pack(fill=tk.BOTH, expand=True)
            new_btn.config(bg=W98['bg'], relief=tk.RAISED)

    def get_frame(self, idx):
        return self.tabs[idx][1]


# ─────────────────────────────────────────────────────────────────────────────
# DZIAŁAJĄCE MENU (Menubar)
# ─────────────────────────────────────────────────────────────────────────────

class W98Menubar:
    def __init__(self, parent, app):
        self.app = app
        self.bar = tk.Frame(parent, bg=W98['bg'], relief=tk.RAISED, bd=1)
        self.bar.pack(fill=tk.X)
        self._open_menu = None
        self._build()

    def _build(self):
        menus = {
            "Plik": [
                ("📂  Wczytaj CSV...", self.app.load_data),
                ("💾  Eksportuj wyniki CSV...", self.app.export_results_csv),
                ("📄  Eksportuj wnioski TXT...", self.app.export_observations_txt),
                None,
                ("❌  Wyjdź", self.app.root.quit),
            ],
            "Widok": [
                ("📋  Analiza zbioru", lambda: self.app.notebook.select(0)),
                ("🌿  Konfiguracja gałęzi", lambda: self.app.notebook.select(1)),
                ("📈  Tabela wyników", lambda: self.app.notebook.select(2)),
                ("📊  Wykresy porównawcze", lambda: self.app.notebook.select(3)),
                ("🔬  Krzywa uczenia", lambda: self.app.notebook.select(4)),
                ("🏆  Najlepszy model", lambda: self.app.notebook.select(5)),
                ("📝  Obserwacje i wnioski", lambda: self.app.notebook.select(6)),
                ("📖  Instrukcja i opis", lambda: self.app.notebook.select(7)),
            ],
            "Narzędzia": [
                ("📊  Analiza danych", self.app.analyze_data),
                ("🔍  Wybór cech", self.app.select_features),
                ("▶   Uruchom eksperymenty...", self.app.ask_experiment_count),
                None,
                ("🗑  Wyczyść log", self.app.clear_log),
            ],
            "Pomoc": [
                ("ℹ   O programie", self._show_about),
            ],
        }

        for name, items in menus.items():
            btn = tk.Button(
                self.bar, text=name,
                bg=W98['bg'], fg=W98['text'],
                font=W98['font'],
                relief=tk.FLAT, bd=1,
                padx=6, pady=2,
                activebackground=W98['title_bg'],
                activeforeground='white',
                cursor='arrow')
            btn.pack(side=tk.LEFT)
            btn.bind("<Button-1>", lambda e, b=btn, i=items: self._toggle_menu(b, i))
            btn.bind("<Enter>", lambda e, b=btn, i=items: self._on_hover(b, i))

        self.app.root.bind("<Button-1>", self._close_on_click, add=True)

    def _toggle_menu(self, btn, items):
        if self._open_menu:
            self._open_menu.destroy()
            self._open_menu = None
        self._show_dropdown(btn, items)

    def _on_hover(self, btn, items):
        if self._open_menu:
            self._open_menu.destroy()
            self._open_menu = None
            self._show_dropdown(btn, items)

    def _show_dropdown(self, btn, items):
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height()
        menu = tk.Toplevel(self.app.root)
        menu.wm_overrideredirect(True)
        menu.wm_geometry(f"+{x}+{y}")
        menu.configure(bg=W98['bg'])
        menu.attributes('-topmost', True)

        outer = tk.Frame(menu, bg=W98['bg_dark'], bd=1, relief=tk.SOLID)
        outer.pack()
        inner = tk.Frame(outer, bg=W98['bg'], relief=tk.RAISED, bd=1)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        for item in items:
            if item is None:
                tk.Frame(inner, bg=W98['bg_dark'], height=1).pack(fill=tk.X, padx=2, pady=1)
            else:
                label, cmd = item
                row = tk.Frame(inner, bg=W98['bg'], cursor='arrow')
                row.pack(fill=tk.X)
                lbl = tk.Label(row, text=label, bg=W98['bg'], fg=W98['text'],
                               font=W98['font'], anchor=tk.W, padx=14, pady=3)
                lbl.pack(fill=tk.X)

                def _on_enter(e, r=row, l=lbl):
                    r.config(bg=W98['title_bg'])
                    l.config(bg=W98['title_bg'], fg='white')

                def _on_leave(e, r=row, l=lbl):
                    r.config(bg=W98['bg'])
                    l.config(bg=W98['bg'], fg=W98['text'])

                def _on_click(e, c=cmd):
                    self._close()
                    c()

                for w in (row, lbl):
                    w.bind("<Enter>", _on_enter)
                    w.bind("<Leave>", _on_leave)
                    w.bind("<Button-1>", _on_click)

        self._open_menu = menu

    def _close(self):
        if self._open_menu:
            self._open_menu.destroy()
            self._open_menu = None

    def _close_on_click(self, event):
        if self._open_menu and event.widget not in (self._open_menu,):
            try:
                if not str(event.widget).startswith(str(self._open_menu)):
                    self._close()
            except Exception:
                self._close()

    def _show_about(self):
        win = tk.Toplevel(self.app.root)
        win.title("O programie")
        win.geometry("340x180")
        win.configure(bg=W98['bg'])
        win.resizable(False, False)
        win.grab_set()
        w98_title_bar(win, "O programie", icon="📊")
        tk.Label(win, text="\nŚrodowisko Eksperymentów ML\n",
                 bg=W98['bg'], fg=W98['text'], font=W98['font_title']).pack()
        tk.Label(win, text="DecisionTreeClassifier · scikit-learn\n"
                           "SelectKBest · f_classif\n"
                           "Obsługuje zbiory ≥1 mln rekordów",
                 bg=W98['bg'], fg=W98['text'], font=W98['font']).pack()
        w98_button(win, "OK", win.destroy, width=10).pack(pady=10)


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNA APLIKACJA
# ─────────────────────────────────────────────────────────────────────────────

class MLProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Środowisko Eksperymentów ML")
        self.root.geometry("1280x860")
        self.root.configure(bg=W98['bg'])
        self.root.resizable(True, True)

        self.df = None
        self.temp_raw_df = None
        self.branches = {}
        self.all_results = []
        self.feature_scores = {}
        self._meta = {}

        self._apply_ttk_style()
        self.create_widgets()

    def _apply_ttk_style(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("W98.Treeview",
                        background=W98['bg_light'], foreground=W98['text'],
                        rowheight=18, fieldbackground=W98['bg_light'],
                        font=W98['font'])
        style.configure("W98.Treeview.Heading",
                        background=W98['bg'], foreground=W98['text'],
                        font=W98['font_bold'], relief=tk.RAISED)
        style.map("W98.Treeview",
                  background=[('selected', W98['select_bg'])],
                  foreground=[('selected', W98['select_fg'])])
        style.configure("W98.Vertical.TScrollbar", background=W98['bg'])
        style.configure("W98.Horizontal.TScrollbar", background=W98['bg'])
        style.configure("W98.Horizontal.TProgressbar",
                        troughcolor=W98['bg_light'],
                        background=W98['title_bg'],
                        thickness=14)

    # ─────────────────────────────────────────────────────────────────────
    # BUDOWANIE UI
    # ─────────────────────────────────────────────────────────────────────

    def create_widgets(self):
        w98_title_bar(self.root, "Środowisko Eksperymentów ML — Projekt", icon="📊")

        self.menubar = W98Menubar(self.root, self)
        w98_separator(self.root)

        main = tk.Frame(self.root, bg=W98['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        self.sidebar = tk.Frame(main, bg=W98['bg'], width=200, relief=tk.GROOVE, bd=2)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.sidebar.pack_propagate(False)

        right = tk.Frame(main, bg=W98['bg'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.notebook = W98Notebook(right)

        prog_f = tk.Frame(right, bg=W98['bg'])
        prog_f.pack(fill=tk.X, pady=(2, 0))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(prog_f, variable=self.progress_var,
                                            maximum=100, style="W98.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, padx=2)
        self.progress_lbl = tk.Label(prog_f, text="", bg=W98['bg'], fg=W98['text'],
                                     font=W98['font'])
        self.progress_lbl.pack(anchor=tk.W)

        log_outer, log_inner = w98_labelframe(right, "Konsola / Log")
        log_outer.pack(fill=tk.X, pady=(2, 0))
        self.console = tk.Text(log_inner, height=5,
                               bg=W98['console_bg'], fg=W98['console_fg'],
                               font=W98['font_mono'], relief=tk.SUNKEN, bd=1,
                               insertbackground='white', state=tk.NORMAL)
        csb = w98_scrollbar(log_inner, command=self.console.yview)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.configure(yscrollcommand=csb.set)
        self.console.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.build_sidebar()
        self.build_tabs()

        self.statusbar = tk.Label(self.root,
                                  text="  Gotowy. Wczytaj plik CSV aby rozpocząć.",
                                  bg=W98['bg_dark'], fg='white',
                                  font=W98['font'], anchor=tk.W, relief=tk.SUNKEN, bd=1)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.log("System gotowy. Oczekuję na wczytanie danych...")

    def set_status(self, text):
        self.statusbar.config(text=f"  {text}")
        self.root.update()

    def set_progress(self, value, label=""):
        self.progress_var.set(value)
        self.progress_lbl.config(text=label)
        self.root.update()

    def build_sidebar(self):
        hdr = tk.Frame(self.sidebar, bg=W98['title_bg'])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="  📁 ML Eksperymenty",
                 bg=W98['title_bg'], fg='white',
                 font=W98['font_bold'], pady=8, anchor=tk.W).pack(fill=tk.X)

        tk.Frame(self.sidebar, bg=W98['bg'], height=8).pack()

        steps = [
            ("📂  1. Wczytaj Zbiór", self.load_data),
            ("📊  2. Analiza Danych", self.analyze_data),
            ("🔍  3. Wybór Cech", self.select_features),
            ("▶   4. Uruchom\n       Eksperymenty", self.ask_experiment_count),
        ]
        for label, cmd in steps:
            btn = w98_button(self.sidebar, label, cmd)
            btn.config(anchor=tk.W, padx=8, pady=4, wraplength=180, justify=tk.LEFT)
            btn.pack(fill=tk.X, padx=6, pady=3)

        tk.Frame(self.sidebar, bg=W98['bg_dark'], height=1).pack(fill=tk.X, padx=4, pady=8)
        tk.Frame(self.sidebar, bg='white', height=1).pack(fill=tk.X, padx=4)
        tk.Frame(self.sidebar, bg=W98['bg'], height=4).pack()

        w98_label(self.sidebar, "Eksport:", bold=True).pack(anchor=tk.W, padx=6, pady=(4, 0))
        w98_button(self.sidebar, "💾 Wyniki → CSV", self.export_results_csv).pack(
            fill=tk.X, padx=6, pady=2)
        w98_button(self.sidebar, "📄 Wnioski → TXT", self.export_observations_txt).pack(
            fill=tk.X, padx=6, pady=2)

        tk.Frame(self.sidebar, bg=W98['bg'], height=8).pack()
        tk.Label(self.sidebar,
                 text="DecisionTreeClassifier\nscikit-learn",
                 bg=W98['bg'], fg=W98['disabled'],
                 font=('Courier New', 7), justify=tk.CENTER).pack(pady=4)

    def build_tabs(self):
        t1 = self.notebook.add("Analiza Zbioru")
        self._build_tab_data(t1)

        t2 = self.notebook.add("Konfiguracja Gałęzi")
        self._build_tab_branches(t2)

        t3 = self.notebook.add("Tabela Wyników")
        self._build_tab_table(t3)

        t4 = self.notebook.add("Wykresy Porównawcze")
        self.compare_chart_frame = tk.Frame(t4, bg=W98['bg_light'])
        self.compare_chart_frame.pack(fill=tk.BOTH, expand=True)

        t5 = self.notebook.add("Krzywa Uczenia")
        self.learning_curve_frame = tk.Frame(t5, bg=W98['bg_light'])
        self.learning_curve_frame.pack(fill=tk.BOTH, expand=True)

        t6 = self.notebook.add("Najlepszy Model")
        self._build_tab_results(t6)

        t7 = self.notebook.add("Obserwacje i Wnioski")
        self.build_observations_tab(t7)

        t8 = self.notebook.add("Instrukcja")
        self._build_tab_instruction(t8)

    def _build_tab_data(self, parent):
        w98_title_bar(parent, "Informacje o wczytanym zbiorze danych", icon="📋")

        info_f = tk.Frame(parent, bg=W98['bg'])
        info_f.pack(fill=tk.X, padx=8, pady=6)

        labels_def = [
            ("Liczba rekordów:", "lbl_records"),
            ("Liczba cech:", "lbl_features"),
            ("Typy kolumn:", "lbl_types"),
            ("Brakujące wartości:", "lbl_missing"),
            ("Zbalansowanie klas:", "lbl_balance"),
            ("Kolumna docelowa:", "lbl_target"),
        ]
        for i, (caption, attr) in enumerate(labels_def):
            tk.Label(info_f, text=caption, bg=W98['bg'], fg=W98['text'],
                     font=W98['font_bold'], width=22, anchor=tk.E).grid(
                row=i, column=0, sticky=tk.E, padx=(0, 4), pady=1)
            lbl = tk.Label(info_f, text="—", bg=W98['bg'], fg=W98['text'],
                           font=W98['font'], anchor=tk.W)
            lbl.grid(row=i, column=1, sticky=tk.W, pady=1)
            setattr(self, attr, lbl)

        w98_separator(parent)
        w98_label(parent, "Statystyki opisowe cech:", bold=True).pack(
            anchor=tk.W, padx=8, pady=(4, 2))

        stats_wrap = tk.Frame(parent, bg=W98['bg'])
        stats_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        self.stats_tree = ttk.Treeview(stats_wrap, style="W98.Treeview", height=7)
        sy = ttk.Scrollbar(stats_wrap, orient=tk.VERTICAL, command=self.stats_tree.yview,
                           style="W98.Vertical.TScrollbar")
        sx = ttk.Scrollbar(stats_wrap, orient=tk.HORIZONTAL, command=self.stats_tree.xview,
                           style="W98.Horizontal.TScrollbar")
        self.stats_tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        self.stats_tree.pack(fill=tk.BOTH, expand=True)

        _, self.class_chart_frame = w98_labelframe(parent, "Rozkład klas (cel klasyfikacji)")
        self.class_chart_frame.master.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _build_tab_branches(self, parent):
        w98_title_bar(parent, "Zestawy cech — 3 gałęzie eksperymentów", icon="🌿")

        branches_f = tk.Frame(parent, bg=W98['bg'])
        branches_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.listboxes = []
        titles = [
            "Gałąź 1: Wszystkie cechy",
            "Gałąź 2: Top 50% (SelectKBest)",
            "Gałąź 3: Top 20% (ścisła selekcja)",
        ]
        for t in titles:
            col = tk.Frame(branches_f, bg=W98['bg'])
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            w98_label(col, t, bold=True).pack(anchor=tk.W, pady=(0, 2))
            box_f = tk.Frame(col, bg=W98['bg'], relief=tk.SUNKEN, bd=2)
            box_f.pack(fill=tk.BOTH, expand=True)
            sb = w98_scrollbar(box_f)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb = w98_listbox(box_f, yscrollcommand=sb.set)
            lb.pack(fill=tk.BOTH, expand=True)
            sb.config(command=lb.yview)
            self.listboxes.append(lb)

        _, self.feat_chart_frame = w98_labelframe(
            parent, "Ważność cech — SelectKBest (f_classif score)")
        self.feat_chart_frame.master.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _build_tab_table(self, parent):
        w98_title_bar(parent, "Wyniki wszystkich eksperymentów", icon="📈")

        tbl_f = tk.Frame(parent, bg=W98['bg'])
        tbl_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        cols = ("Nr", "Gałąź", "max_depth", "Accuracy", "Precision",
                "Recall", "F1", "Train Acc", "Overfit?")
        self.results_tree = ttk.Treeview(tbl_f, columns=cols, show='headings',
                                         style="W98.Treeview", height=25)
        widths = [35, 150, 75, 85, 85, 85, 85, 85, 75]
        for c, w in zip(cols, widths):
            self.results_tree.heading(c, text=c)
            self.results_tree.column(c, width=w, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(tbl_f, orient=tk.VERTICAL, command=self.results_tree.yview,
                            style="W98.Vertical.TScrollbar")
        self.results_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.pack(fill=tk.BOTH, expand=True)

        self.results_tree.tag_configure('best', background='#00ff00', foreground='#000080')
        self.results_tree.tag_configure('overfit', background='#ffdddd')
        self.results_tree.tag_configure('branch1', background='#ffffff')
        self.results_tree.tag_configure('branch2', background='#eeeeee')
        self.results_tree.tag_configure('branch3', background='#dde8ff')

    def _build_tab_results(self, parent):
        w98_title_bar(parent, "Najlepszy model spośród wszystkich eksperymentów", icon="🏆")

        mf = tk.Frame(parent, bg=W98['bg'])
        mf.pack(fill=tk.X, padx=8, pady=8)

        self.lbl_best_info = tk.Label(mf, text="Mierniki najlepszego modelu:",
                                      bg=W98['bg'], fg=W98['text'], font=W98['font_title'])
        self.lbl_best_info.grid(row=0, column=0, columnspan=5, sticky=tk.W, pady=(0, 6))

        for col, (attr, txt) in enumerate([
            ('lbl_acc', "Accuracy: —"),
            ('lbl_prec', "Precision: —"),
            ('lbl_rec', "Recall: —"),
            ('lbl_f1', "F1-Score: —"),
            ('lbl_cv', "CV Score: —"),
        ]):
            lbl = tk.Label(mf, text=txt, bg=W98['bg'], fg=W98['text'],
                           font=W98['font'], relief=tk.GROOVE, bd=1, padx=8, pady=4)
            lbl.grid(row=1, column=col, sticky=tk.W, padx=6, pady=2)
            setattr(self, attr, lbl)

        split = tk.Frame(parent, bg=W98['bg'])
        split.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        _, self.cm_frame = w98_labelframe(split, "Macierz Pomyłek (Confusion Matrix)")
        self.cm_frame.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        _, self.tree_frame = w98_labelframe(split, "Wizualizacja Drzewa Decyzyjnego (głębokość ≤3)")
        self.tree_frame.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def build_observations_tab(self, parent):
        w98_title_bar(parent, "Obserwacje i wnioski z eksperymentów", icon="📝")

        paned = tk.Frame(parent, bg=W98['bg'])
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(paned, bg=W98['bg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        w98_label(left, "Automatyczne spostrzeżenia:", bold=True).pack(anchor=tk.W, pady=(0, 2))
        auto_f = tk.Frame(left, bg=W98['console_bg'], relief=tk.SUNKEN, bd=2)
        auto_f.pack(fill=tk.BOTH, expand=True)
        asb = w98_scrollbar(auto_f)
        asb.pack(side=tk.RIGHT, fill=tk.Y)
        self.auto_obs_text = tk.Text(auto_f,
                                     bg=W98['console_bg'], fg='#00ff00',
                                     font=W98['font_mono'], wrap=tk.WORD,
                                     state=tk.DISABLED, relief=tk.FLAT,
                                     yscrollcommand=asb.set)
        asb.config(command=self.auto_obs_text.yview)
        self.auto_obs_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        right = tk.Frame(paned, bg=W98['bg'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sections = [
            ("Obserwacje (wpisz swoje spostrzeżenia):", "obs_text"),
            ("Wnioski (na podstawie obserwacji):", "wn_text"),
            ("Wnioski końcowe do projektu:", "end_text"),
        ]
        for caption, attr in sections:
            w98_label(right, caption, bold=True).pack(anchor=tk.W, pady=(6, 1))
            f = tk.Frame(right, bg=W98['bg_light'], relief=tk.SUNKEN, bd=2)
            f.pack(fill=tk.BOTH, expand=True)
            sb2 = w98_scrollbar(f)
            sb2.pack(side=tk.RIGHT, fill=tk.Y)
            txt = tk.Text(f, bg=W98['bg_light'], fg=W98['text'],
                          font=W98['font'], wrap=tk.WORD, height=5,
                          relief=tk.FLAT, yscrollcommand=sb2.set)
            sb2.config(command=txt.yview)
            txt.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            setattr(self, attr, txt)

    def _build_tab_instruction(self, parent):
        w98_title_bar(parent, "Instrukcja obsługi środowiska i interpretacji wyników", icon="📖")

        f = tk.Frame(parent, bg=W98['bg_light'], relief=tk.SUNKEN, bd=2)
        f.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        sb = w98_scrollbar(f)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        txt = tk.Text(f, bg=W98['bg_light'], fg=W98['text'],
                      font=W98['font_mono'], wrap=tk.WORD,
                      relief=tk.FLAT, yscrollcommand=sb.set)
        sb.config(command=txt.yview)
        txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        instrukcja_tresc = """================================================================================
  ŚRODOWISKO EKSPERYMENTÓW ML - MANUAL I DOKUMENTACJA SYSTEMOWA
================================================================================

1. OPIS ZMIENNYCH I MIERNIKÓW JAKOŚCI (METRYK)
--------------------------------------------------------------------------------
* max_depth (Maksymalna głębokość):
  Główny hiperparametr drzewa decyzyjnego. Określa limit poziomów (podziałów) 
  drzewa. 
  - Niska wartość (np. 1-3) upraszcza model, grożąc niedouczeniem (underfitting).
  - Wysoka wartość pozwala dopasować się do detali, grożąc przeuczeniem (overfitting).

* Accuracy (Dokładność):
  Stosunek poprawnie sklasyfikowanych próbek do wszystkich próbek w zbiorze. 
  Główna miara sukcesu, gdy klasy w zbiorze są zbalansowane.

* Precision (Precyzja):
  Miara mówiąca o tym, jak wiele z próbek wskazanych przez model jako pozytywne,
  jest nimi w rzeczywistości. Kluczowa, gdy fałszywy alarm (False Positive) 
  generuje duże koszty lub problemy.

* Recall (Czułość / Pełność):
  Zdolność modelu do wykrywania wszystkich rzeczywistych próbek danej klasy. 
  Kluczowa w obszarach medycznych lub detekcji awarii, gdzie przeoczenie obiektu
  pozytywnego (False Negative) ma poważne skutki.

* F1-Score:
  Średnia harmoniczna miar Precision i Recall. Daje obiektywny, zbalansowany obraz
  skuteczności klasyfikatora w przypadku nierównego rozkładu klas (target balance).

* Train Acc (Dokładność na zbiorze treningowym):
  Wynik osiągany na danych, które model widział podczas nauki. Porównanie tej wartości
  z Accuracy (testowym) pozwala wykryć, czy algorytm nie uczy się struktur na pamięć.

* Overfit? (Przeuczenie):
  Flaga ostrzegawcza (⚠ TAK). Zapala się automatycznie, gdy różnica między 
  skutecznością treningową a testową przekracza 5% punktów procentowych.

* CV Score (Walidacja krzyżowa 5-fold):
  Uśredniony wynik dokładności wyznaczony poprzez 5-krotny niezależny podział 
  danych. Gwarantuje stabilność oceny i chroni przed losowym "szczęśliwym podziałem" 
  bazy testowej.


2. ARCHITEKTURA GAŁĘZI EKSPERYMENTÓW (REDUKCJA WYMIAROWOŚCI)
--------------------------------------------------------------------------------
Aplikacja bada zachowanie algorytmu w trzech scenariuszach wyboru cech:
* Gałąź 1: Model otrzymuje 100% dostępnych w bazie cech numerycznych.
* Gałąź 2: Selekcja SelectKBest odrzuca najmniej powiązane zmienne, zostawiając 50%.
* Gałąź 3: Ścisła selekcja zachowuje tylko 20% cech o najwyższym wyniku ANOVA F-value.


3. INTERPRETACJA I DZIAŁANIE WYKRESÓW GRAFICZNYCH
--------------------------------------------------------------------------------
* Wykres "Rozkład klas" (Karta: Analiza Zbioru):
  Wykres słupkowy prezentujący udziały procentowe klas decyzyjnych. Jeśli słupki 
  są skrajnie nierówne, należy odrzucić miarę Accuracy na rzecz metryki F1-Score.

* Wykres "Ważność cech — SelectKBest" (Karta: Konfiguracja Gałęzi):
  Horyzontalny wykres słupkowy. Długość paska odzwierciedla wynik testu f_classif. 
  Cechy z samej góry mają największy wpływ na poprawność predykcji.

* Wykresy Liniowe Metryk (Karta: Wykresy Porównawcze):
  Cztery górne wykresy ilustrują zachowanie miar Acc, Prec, Rec, F1 w funkcji głębokości 
  drzewa. Pozwalają zaobserwować, od którego momentu parametry przestają rosnąć.

* Wykres "Train vs Test Accuracy" (Karta: Wykresy Porównawcze):
  Kluczowe narzędzie diagnostyczne. Linia ciągła (Test) zazwyczaj rośnie do pewnego 
  punktu, po czym zaczyna opadać. Linia przerywana (Train) dąży do 100%. Moment, 
  w którym linie zaczynają się od siebie drastycznie oddalać, to punkt krytyczny 
  wejścia w strefę przeuczenia (Overfitting).

* Wykres "Średnia Train vs Test per gałąź" (Karta: Wykresy Porównawcze):
  Zbiorcze porównanie stabilności gałęzi. Pokazuje, czy redukcja cech (G2 i G3) 
  zmniejszyła różnicę między nauką a testem, stabilizując działanie modelu.

* Wykresy "Krzywa Uczenia" (Karta: Krzywa Uczenia):
  - Lewy: Pokazuje spadek błędu i stabilizację metryk w miarę dopływu nowych 
    wierszy danych do treningu. Czerwone pole obrazuje lukę przeuczenia.
  - Prawy: Profiluje optymalne max_depth na wybranej gałęzi wskazując zieloną linią 
    punkt maksymalnej efektywności testowej.

* Wykres "Confusion Matrix" (Karta: Najlepszy Model):
  Macierz pomyłek weryfikująca strukturę błędów. Ukątna (od lewego górnego do prawego 
  dolnego rogu) grupuje trafienia bezbłędne. Pozostałe pola to błędy klasyfikacji.

* Wykres "Wizualizacja Drzewa" (Karta: Najlepszy Model):
  Rysunek struktury decyzyjnej do poziomu głębokości 3. Pozwala prześledzić kryteria 
  oraz progi logiczne, jakimi kierował się zwycięski model podczas podziału danych.
================================================================================"""

        txt.insert(tk.END, instrukcja_tresc)
        txt.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────────────────────────────────

    def log(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"> {message}\n")
        self.console.see(tk.END)
        self.root.update()

    def clear_log(self):
        self.console.config(state=tk.NORMAL)
        self.console.delete('1.0', tk.END)

    # ─────────────────────────────────────────────────────────────────────
    # WCZYTYWANIE DANYCH
    # ─────────────────────────────────────────────────────────────────────

    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath:
            return
        self.set_status(f"Wczytywanie: {filepath}")
        self.log(f"Wczytywanie: {filepath} ...")
        try:
            self.temp_raw_df = pd.read_csv(filepath)
            self.log(f"Wczytano — {self.temp_raw_df.shape[0]} wierszy, "
                     f"{self.temp_raw_df.shape[1]} kolumn.")
            self.show_column_selector()
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            self.log(f"Błąd: {e}")

    def show_column_selector(self):
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Zarządzanie Kolumnami")
        self.popup.geometry("460x500")
        self.popup.configure(bg=W98['bg'])
        self.popup.grab_set()
        self.popup.resizable(False, False)

        w98_title_bar(self.popup, "Wybierz kolumny do usunięcia", icon="🗂")
        w98_label(self.popup,
                  "Zaznacz kolumny, które chcesz ZIGNOROWAĆ (usunąć).", bold=True).pack(
            pady=(8, 2), padx=10, anchor=tk.W)
        w98_label(self.popup,
                  "Ostatnia niezaznaczona kolumna = kolumna docelowa (Target).").pack(
            padx=10, anchor=tk.W)

        frame = tk.Frame(self.popup, bg=W98['bg_light'], relief=tk.SUNKEN, bd=2)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        sb = w98_scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.col_listbox = w98_listbox(frame, selectmode=tk.MULTIPLE, yscrollcommand=sb.set)
        for col in self.temp_raw_df.columns:
            self.col_listbox.insert(tk.END, col)
        self.col_listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.col_listbox.yview)

        w98_separator(self.popup)
        bf = tk.Frame(self.popup, bg=W98['bg'])
        bf.pack(fill=tk.X, padx=10, pady=8)
        w98_button(bf, "✔  Zatwierdź", self.apply_column_selection, width=16).pack(
            side=tk.LEFT, padx=(0, 8))
        w98_button(bf, "✖  Anuluj", self._cancel_load, width=16).pack(side=tk.LEFT)

    def _cancel_load(self):
        self.temp_raw_df = None
        self.popup.destroy()
        self.log("Anulowano wczytywanie.")
        self.set_status("Gotowy.")

    def apply_column_selection(self):
        sel = self.col_listbox.curselection()
        to_drop = [self.col_listbox.get(i) for i in sel]
        if to_drop:
            self.temp_raw_df.drop(columns=to_drop, inplace=True)
            self.log(f"Usunięto kolumny: {', '.join(to_drop)}")

        missing_before = self.temp_raw_df.isnull().sum().sum()
        numeric_before = self.temp_raw_df.select_dtypes(include=[np.number]).shape[1]
        object_before = self.temp_raw_df.select_dtypes(include=['object']).shape[1]

        self.df = self.temp_raw_df.select_dtypes(include=[np.number]).dropna()

        if len(self.df) > 1_200_000:
            self.log("Próbkowanie do 1.2 mln rekordów (ochrona RAM)...")
            self.df = self.df.sample(n=1_200_000, random_state=42)

        if self.df.empty or len(self.df.columns) < 2:
            self.log("BŁĄD: Zbiór jest pusty lub za mało kolumn!")
            self.df = None
        else:
            r, c = self.df.shape
            target = self.df.columns[-1]
            self._meta = {
                'rows': r, 'features': c - 1, 'target': target,
                'missing': missing_before,
                'numeric': numeric_before, 'object': object_before
            }
            self.log(f"Zbiór gotowy: {r:,} rekordów, {c - 1} cech + cel '{target}'.")
            self.set_status(f"Wczytano: {r:,} rekordów, {c - 1} cech, cel: '{target}'")

        self.temp_raw_df = None
        self.popup.destroy()

    # ─────────────────────────────────────────────────────────────────────
    # ANALIZA DANYCH
    # ─────────────────────────────────────────────────────────────────────

    def analyze_data(self):
        if self.df is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane (krok 1)!")
            return

        m = self._meta
        target_col = m['target']
        class_counts = self.df[target_col].value_counts()
        class_pct = class_counts / len(self.df) * 100

        self.lbl_records.config(text=f"{m['rows']:,}")
        self.lbl_features.config(text=f"{m['features']}")
        self.lbl_types.config(text=f"numeryczne: {m['numeric']}, "
                                   f"tekstowe (wykluczone): {m['object']}")
        self.lbl_missing.config(text=f"{m['missing']:,}")
        self.lbl_target.config(text=f"'{target_col}'  ({len(class_counts)} klas)")
        self.lbl_balance.config(
            text=",  ".join([f"'{k}': {v:.1f}%" for k, v in class_pct.items()]))

        stats = self.df.iloc[:, :-1].describe().T[
            ['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(4)
        self.stats_tree.delete(*self.stats_tree.get_children())
        cols = ['Cecha'] + list(stats.columns)
        self.stats_tree['columns'] = cols
        self.stats_tree['show'] = 'headings'
        self.stats_tree.heading('Cecha', text='Cecha')
        self.stats_tree.column('Cecha', width=160, anchor=tk.W)
        for c in stats.columns:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=85, anchor=tk.CENTER)
        for feat, row in stats.iterrows():
            self.stats_tree.insert('', tk.END, values=[feat] + list(row))

        for w in self.class_chart_frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(6, 2.2))
        fig.patch.set_facecolor('#c0c0c0')
        ax.set_facecolor('#ffffff')
        colors = ['#000080', '#808080', '#c0c0c0', '#008080', '#800000']
        bars = ax.bar([str(k) for k in class_counts.index],
                      class_pct.values,
                      color=colors[:len(class_counts)], edgecolor='black', linewidth=0.8)
        ax.set_title("Rozkład klas", fontsize=9)
        ax.set_ylabel("%", fontsize=8)
        for bar, pct in zip(bars, class_pct.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.class_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

        if any(v < 30.0 for v in class_pct.values):
            self.log("OSTRZEŻENIE: Niezbalansowanie klas! "
                     "Zwróć uwagę na Recall i Precision.")

        self.log("Analiza zakończona.")
        self.set_status("Analiza danych zakończona.")
        self.notebook.select(0)

    # ─────────────────────────────────────────────────────────────────────
    # WYBÓR CECH
    # ─────────────────────────────────────────────────────────────────────

    def select_features(self):
        if self.df is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj dane (krok 1)!")
            return

        self.log("Selekcja cech — SelectKBest / f_classif ...")
        self.set_status("Obliczanie ważności cech...")

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        n = len(X.columns)

        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        scores = selector.scores_
        self.feature_scores = dict(zip(X.columns, scores))

        self.branches['Gałąź 1'] = list(X.columns)

        k50 = max(1, n // 2)
        top50_idx = np.argsort(scores)[::-1][:k50]
        self.branches['Gałąź 2'] = list(X.columns[np.sort(top50_idx)])

        k20 = max(1, int(n * 0.2))
        top20_idx = np.argsort(scores)[::-1][:k20]
        self.branches['Gałąź 3'] = list(X.columns[np.sort(top20_idx)])

        for i, (_, feats) in enumerate(self.branches.items()):
            self.listboxes[i].delete(0, tk.END)
            for f in feats:
                self.listboxes[i].insert(tk.END, f)

        self.log(f"Gałąź 1: {len(self.branches['Gałąź 1'])} | "
                 f"Gałąź 2: {len(self.branches['Gałąź 2'])} | "
                 f"Gałąź 3: {len(self.branches['Gałąź 3'])} cech.")

        self._draw_feature_importance()
        self.set_status("Selekcja cech zakończona — 3 gałęzie gotowe.")
        self.notebook.select(1)

    def _draw_feature_importance(self):
        for w in self.feat_chart_frame.winfo_children():
            w.destroy()

        sorted_feats = sorted(self.feature_scores.items(),
                              key=lambda x: x[1], reverse=True)[:20]
        names = [f[0] for f in sorted_feats]
        values = [f[1] for f in sorted_feats]

        fig, ax = plt.subplots(figsize=(8, max(2.5, len(names) * 0.3)))
        fig.patch.set_facecolor('#c0c0c0')
        ax.set_facecolor('#ffffff')
        colors = ['#000080' if i < max(1, len(names) // 5) else
                  '#808080' if i < len(names) // 2 else '#c0c0c0'
                  for i in range(len(names))]
        ax.barh(names[::-1], values[::-1], color=colors[::-1],
                edgecolor='black', linewidth=0.5)
        ax.set_title("Ważność cech — SelectKBest (f_classif)", fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.feat_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # OKIENKO LICZBY EKSPERYMENTÓW
    # ─────────────────────────────────────────────────────────────────────

    def ask_experiment_count(self):
        if not self.branches:
            messagebox.showwarning("Uwaga", "Najpierw wykonaj wybór cech (krok 3)!")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Uruchom eksperymenty")
        popup.geometry("400x300")
        popup.configure(bg=W98['bg'])
        popup.resizable(False, False)
        popup.grab_set()

        w98_title_bar(popup, "Konfiguracja eksperymentów", icon="⚙")

        cf = tk.Frame(popup, bg=W98['bg'])
        cf.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        w98_label(cf, "Podaj łączną liczbę eksperymentów (min. 30).\n"
                      "Zostaną równo podzielone na 3 gałęzie.").pack(anchor=tk.W, pady=(0, 8))

        row = tk.Frame(cf, bg=W98['bg'])
        row.pack(fill=tk.X)
        w98_label(row, "Liczba eksperymentów:").pack(side=tk.LEFT)
        var = tk.StringVar(value="30")
        vcmd = (self.root.register(lambda v: v == "" or v.isdigit()), '%P')
        entry = w98_entry(row, textvariable=var, width=7,
                          validate='key', validatecommand=vcmd)
        entry.pack(side=tk.LEFT, padx=8)

        lbl_hint = tk.Label(cf, text="Na gałąź: ~10", bg=W98['bg'],
                            fg=W98['disabled'], font=W98['font'])
        lbl_hint.pack(anchor=tk.W, pady=(4, 0))

        # Walidacja krzyżowa
        cv_var = tk.BooleanVar(value=True)
        cv_row = tk.Frame(cf, bg=W98['bg'])
        cv_row.pack(fill=tk.X, pady=(8, 0))
        tk.Checkbutton(cv_row, text="Uruchom walidację krzyżową (5-fold CV) dla najlepszego modelu",
                       variable=cv_var, bg=W98['bg'], fg=W98['text'],
                       font=W98['font'], activebackground=W98['bg']).pack(anchor=tk.W)

        def update_hint(*_):
            try:
                lbl_hint.config(text=f"Na gałąź: ~{max(1, int(var.get()) // 3)}")
            except ValueError:
                lbl_hint.config(text="Na gałąź: —")

        var.trace_add("write", update_hint)

        w98_separator(cf)

        def confirm():
            raw = var.get().strip()
            if not raw or not raw.isdigit() or int(raw) < 1:
                messagebox.showwarning("Błąd", "Podaj liczbę całkowitą > 0.", parent=popup)
                return
            do_cv = cv_var.get()
            popup.destroy()
            self.run_experiments(int(raw), do_cv=do_cv)

        bf = tk.Frame(cf, bg=W98['bg'])
        bf.pack(fill=tk.X, pady=(10, 0))
        w98_button(bf, "▶  Uruchom", confirm, width=14).pack(side=tk.LEFT, padx=(0, 8))
        w98_button(bf, "Anuluj", popup.destroy, width=10).pack(side=tk.LEFT)

        entry.focus_set()
        entry.select_range(0, tk.END)
        popup.bind("<Return>", lambda e: confirm())
        popup.bind("<Escape>", lambda e: popup.destroy())

    # ─────────────────────────────────────────────────────────────────────
    # EKSPERYMENTY
    # ─────────────────────────────────────────────────────────────────────

    def run_experiments(self, total=30, do_cv=True):
        n_per_branch = max(1, total // 3)
        actual_total = n_per_branch * 3

        self.log(f"Start: {actual_total} eksperymentów ({n_per_branch}/gałąź × 3)...")
        self.set_status(f"Trwa {actual_total} eksperymentów...")
        self.all_results = []
        self.results_tree.delete(*self.results_tree.get_children())

        y = self.df.iloc[:, -1]

        if n_per_branch == 1:
            depths = [5]
        else:
            depths = sorted(set(np.linspace(1, 30, n_per_branch, dtype=int).tolist()))[:n_per_branch]

        exp_nr = 1
        best_acc = -1
        best_result = None
        best_cm = None
        best_model = None
        best_features = None

        for branch_name, features in self.branches.items():
            X = self.df[features]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)

            for depth in depths:
                pct = (exp_nr - 1) / actual_total * 100
                self.set_progress(pct, f"Eksperyment {exp_nr}/{actual_total} — "
                                       f"{branch_name}, depth={int(depth)}")

                model = DecisionTreeClassifier(max_depth=int(depth), random_state=42)
                model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                train_acc = accuracy_score(y_train, y_pred_train)
                acc = accuracy_score(y_test, y_pred_test)
                prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                overfit = (train_acc - acc) > 0.05

                result = {
                    'nr': exp_nr, 'branch': branch_name, 'depth': int(depth),
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                    'train_acc': train_acc, 'overfit': overfit,
                    'classes': model.classes_
                }
                self.all_results.append(result)

                self.log(f"Exp {exp_nr:>3}/{actual_total} | {branch_name} "
                         f"depth={int(depth):>2} | "
                         f"Train={train_acc * 100:.1f}% Test={acc * 100:.1f}% "
                         f"{'⚠OVERFIT' if overfit else ''}  "
                         f"Prec={prec * 100:.1f}% Rec={rec * 100:.1f}% F1={f1 * 100:.1f}%")

                if acc > best_acc:
                    best_acc = acc
                    best_result = result
                    best_cm = confusion_matrix(y_test, y_pred_test)
                    best_model = model
                    best_features = features

                exp_nr += 1

        self.set_progress(100, "Zakończono!")
        self.log(f"=== ZAKOŃCZONO {actual_total} EKSPERYMENTÓW ===")
        self.log(f"Najlepszy: {best_result['branch']} depth={best_result['depth']} "
                 f"Acc={best_result['acc'] * 100:.2f}%")
        self.set_status(f"Zakończono! Najlepszy: {best_result['branch']}, "
                        f"depth={best_result['depth']}, "
                        f"Acc={best_result['acc'] * 100:.2f}%")

        # Walidacja krzyżowa dla najlepszego modelu
        cv_score = None
        if do_cv:
            self.log("Walidacja krzyżowa (5-fold) dla najlepszego modelu...")
            self.set_progress(100, "Cross-validation...")
            try:
                cv_model = DecisionTreeClassifier(
                    max_depth=best_result['depth'], random_state=42)
                X_best = self.df[best_features]
                y_best = self.df.iloc[:, -1]
                if len(X_best) > 100_000:
                    idx = np.random.choice(len(X_best), 100_000, replace=False)
                    X_cv = X_best.iloc[idx]
                    y_cv = y_best.iloc[idx]
                else:
                    X_cv, y_cv = X_best, y_best
                cv_scores = cross_val_score(cv_model, X_cv, y_cv, cv=5, scoring='accuracy')
                cv_score = cv_scores.mean()
                self.log(f"CV (5-fold): {cv_score * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
            except Exception as e:
                self.log(f"CV pominięta: {e}")

        self._fill_results_table(best_result)
        self._draw_comparison_charts()
        self._draw_learning_curve(best_result, best_features)
        self._update_best_model(best_result, best_cm, best_model, best_features, cv_score)
        self._generate_auto_observations()
        self.notebook.select(2)

    # ─────────────────────────────────────────────────────────────────────
    # WYPEŁNIENIE TABELI WYNIKÓW
    # ─────────────────────────────────────────────────────────────────────

    def _fill_results_table(self, best_result):
        self.results_tree.delete(*self.results_tree.get_children())
        tag_map = {'Gałąź 1': 'branch1', 'Gałąź 2': 'branch2', 'Gałąź 3': 'branch3'}
        for r in self.all_results:
            if r['nr'] == best_result['nr']:
                tag = 'best'
            elif r['overfit']:
                tag = 'overfit'
            else:
                tag = tag_map.get(r['branch'], 'branch1')
            self.results_tree.insert('', tk.END, tags=(tag,), values=(
                r['nr'], r['branch'], r['depth'],
                f"{r['acc'] * 100:.2f}%", f"{r['prec'] * 100:.2f}%",
                f"{r['rec'] * 100:.2f}%", f"{r['f1'] * 100:.2f}%",
                f"{r['train_acc'] * 100:.2f}%",
                "⚠ TAK" if r['overfit'] else "OK",
            ))

    # ─────────────────────────────────────────────────────────────────────
    # WYKRESY PORÓWNAWCZE
    # ─────────────────────────────────────────────────────────────────────

    def _draw_comparison_charts(self):
        for w in self.compare_chart_frame.winfo_children():
            w.destroy()

        fig = plt.figure(figsize=(13, 7))
        fig.patch.set_facecolor('#c0c0c0')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

        metrics = [
            ('acc', 'Accuracy (test)'),
            ('prec', 'Precision'),
            ('rec', 'Recall'),
            ('f1', 'F1-Score'),
        ]
        branch_colors = {'Gałąź 1': '#000080', 'Gałąź 2': '#008000', 'Gałąź 3': '#800000'}

        for idx, (mk, mn) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.set_facecolor('#ffffff')
            for bn in self.branches:
                br = [r for r in self.all_results if r['branch'] == bn]
                ax.plot([r['depth'] for r in br], [r[mk] * 100 for r in br],
                        marker='s', markersize=4, label=bn,
                        color=branch_colors.get(bn, 'gray'), linewidth=1.5)
            ax.set_title(mn, fontsize=9, fontweight='bold')
            ax.set_xlabel("max_depth", fontsize=8)
            ax.set_ylabel("[%]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.tick_params(labelsize=7)

        # Dodatkowy wykres
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor('#ffffff')
        for bn, color in branch_colors.items():
            br = sorted([r for r in self.all_results if r['branch'] == bn],
                        key=lambda r: r['depth'])
            ax5.plot([r['depth'] for r in br], [r['train_acc'] * 100 for r in br],
                     linestyle='--', color=color, linewidth=1.2, alpha=0.7,
                     label=f"{bn} (train)")
            ax5.plot([r['depth'] for r in br], [r['acc'] * 100 for r in br],
                     linestyle='-', color=color, linewidth=1.5,
                     label=f"{bn} (test)")
        ax5.set_title("Train vs Test Accuracy", fontsize=9, fontweight='bold')
        ax5.set_xlabel("max_depth", fontsize=8)
        ax5.set_ylabel("[%]", fontsize=8)
        ax5.legend(fontsize=6, ncol=2)
        ax5.grid(True, alpha=0.4, linestyle='--')
        ax5.tick_params(labelsize=7)

        # Wykres słupkowy
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor('#ffffff')
        branch_names = list(self.branches.keys())
        avg_test = [np.mean([r['acc'] * 100 for r in self.all_results if r['branch'] == b])
                    for b in branch_names]
        avg_train = [np.mean([r['train_acc'] * 100 for r in self.all_results if r['branch'] == b])
                     for b in branch_names]
        x = np.arange(len(branch_names))
        ax6.bar(x - 0.2, avg_train, 0.35, label='Train',
                color=['#6699ff', '#66cc66', '#ff9999'], edgecolor='black', linewidth=0.5)
        ax6.bar(x + 0.2, avg_test, 0.35, label='Test',
                color=['#000080', '#008000', '#800000'], edgecolor='black', linewidth=0.5)
        ax6.set_xticks(x)
        ax6.set_xticklabels(['G1', 'G2', 'G3'], fontsize=8)
        ax6.set_title("Śr. Train vs Test per gałąź", fontsize=9, fontweight='bold')
        ax6.set_ylabel("[%]", fontsize=8)
        ax6.legend(fontsize=7)
        ax6.tick_params(labelsize=7)

        fig.suptitle("Wpływ max_depth i zestawu cech na mierniki jakości",
                     fontsize=10, fontweight='bold')

        canvas = FigureCanvasTkAgg(fig, master=self.compare_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # KRZYWA UCZENIA
    # ─────────────────────────────────────────────────────────────────────

    def _draw_learning_curve(self, best_result, best_features):
        for w in self.learning_curve_frame.winfo_children():
            w.destroy()

        self.log("Generowanie krzywej uczenia dla najlepszego modelu...")
        X = self.df[best_features]
        y = self.df.iloc[:, -1]

        if len(X) > 50_000:
            idx = np.random.choice(len(X), 50_000, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]

        train_sizes_pct = np.linspace(0.1, 1.0, 8)
        train_sizes_abs = np.floor(train_sizes_pct * len(X) * 0.7).astype(int)
        train_sizes_abs = np.clip(train_sizes_abs, 10, None)

        train_accs, test_accs = [], []

        for ts in train_sizes_abs:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
            X_tr = X_tr.iloc[:ts]
            y_tr = y_tr.iloc[:ts]
            m = DecisionTreeClassifier(max_depth=best_result['depth'], random_state=42)
            m.fit(X_tr, y_tr)
            train_accs.append(accuracy_score(y_tr, m.predict(X_tr)) * 100)
            test_accs.append(accuracy_score(y_te, m.predict(X_te)) * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#c0c0c0')

        ax1.set_facecolor('#ffffff')
        ax1.plot(train_sizes_abs, train_accs, 'o-', color='#000080',
                 label='Train', linewidth=2, markersize=5)
        ax1.plot(train_sizes_abs, test_accs, 's--', color='#800000',
                 label='Test', linewidth=2, markersize=5)
        ax1.fill_between(train_sizes_abs, train_accs, test_accs, alpha=0.15, color='red',
                         label='Luka overfittingu')
        ax1.set_title(f"Krzywa Uczenia — {best_result['branch']}, "
                      f"depth={best_result['depth']}", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Rozmiar zbioru treningowego", fontsize=9)
        ax1.set_ylabel("Accuracy [%]", fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.4, linestyle='--')

        ax2.set_facecolor('#ffffff')
        best_branch_results = sorted(
            [r for r in self.all_results if r['branch'] == best_result['branch']],
            key=lambda r: r['depth'])
        depths_list = [r['depth'] for r in best_branch_results]
        train_list = [r['train_acc'] * 100 for r in best_branch_results]
        test_list = [r['acc'] * 100 for r in best_branch_results]
        ax2.plot(depths_list, train_list, 'o-', color='#000080',
                 label='Train', linewidth=2, markersize=5)
        ax2.plot(depths_list, test_list, 's--', color='#800000',
                 label='Test', linewidth=2, markersize=5)
        ax2.fill_between(depths_list, train_list, test_list, alpha=0.15, color='red')
        ax2.axvline(x=best_result['depth'], color='green', linestyle=':', linewidth=1.5,
                    label=f"Optymalny depth={best_result['depth']}")
        ax2.set_title(f"Train vs Test wg głębokości — {best_result['branch']}",
                      fontsize=10, fontweight='bold')
        ax2.set_xlabel("max_depth", fontsize=9)
        ax2.set_ylabel("Accuracy [%]", fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.4, linestyle='--')

        fig.suptitle("Analiza overfittingu — krzywe uczenia", fontsize=11, fontweight='bold')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.learning_curve_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
        self.log("Krzywa uczenia wygenerowana.")

    # ─────────────────────────────────────────────────────────────────────
    # NAJLEPSZY MODEL
    # ─────────────────────────────────────────────────────────────────────

    def _update_best_model(self, best, cm, model, features, cv_score=None):
        self.lbl_best_info.config(
            text=f"  Najlepszy: {best['branch']}, max_depth={best['depth']}  ")
        self.lbl_acc.config(text=f"  Accuracy:  {best['acc'] * 100:.2f}%  ")
        self.lbl_prec.config(text=f"  Precision: {best['prec'] * 100:.2f}%  ")
        self.lbl_rec.config(text=f"  Recall:    {best['rec'] * 100:.2f}%  ")
        self.lbl_f1.config(text=f"  F1-Score:  {best['f1'] * 100:.2f}%  ")
        if cv_score is not None:
            self.lbl_cv.config(text=f"  CV (5-fold): {cv_score * 100:.2f}%  ")

        for w in self.cm_frame.winfo_children():
            w.destroy()
        fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
        fig1.patch.set_facecolor('#c0c0c0')
        ax1.set_facecolor('#ffffff')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best['classes'])
        disp.plot(cmap='Blues', ax=ax1, values_format='d')
        ax1.set_title("Macierz Pomyłek", fontsize=9)
        fig1.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig1, master=self.cm_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig1)

        for w in self.tree_frame.winfo_children():
            w.destroy()
        try:
            depth_vis = min(best['depth'], 3)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#c0c0c0')
            ax2.set_facecolor('#ffffff')
            plot_tree(model,
                      max_depth=depth_vis,
                      feature_names=features,
                      class_names=[str(c) for c in best['classes']],
                      filled=True, rounded=True,
                      fontsize=6, ax=ax2,
                      impurity=False, proportion=False)
            ax2.set_title(f"Drzewo decyzyjne (depth ≤ {depth_vis})", fontsize=9)
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=self.tree_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig2)
            self.log("Wizualizacja drzewa wygenerowana.")
        except Exception as e:
            self.log(f"Wizualizacja drzewa niedostępna: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # AUTOMATYCZNE OBSERWACJE
    # ─────────────────────────────────────────────────────────────────────

    def _generate_auto_observations(self):
        if not self.all_results:
            return

        lines = []
        lines.append("=" * 60)
        lines.append("  AUTOMATYCZNE SPOSTRZEZENIA Z EKSPERYMENTOW")
        lines.append("=" * 60)
        lines.append("")

        best = max(self.all_results, key=lambda r: r['acc'])
        worst = min(self.all_results, key=lambda r: r['acc'])

        lines.append(f"[+] Najlepszy: {best['branch']}, depth={best['depth']}")
        lines.append(f"    Acc={best['acc'] * 100:.2f}%  Prec={best['prec'] * 100:.2f}%"
                     f"  Rec={best['rec'] * 100:.2f}%  F1={best['f1'] * 100:.2f}%")
        lines.append(f"    Train={best['train_acc'] * 100:.2f}%  "
                     f"Overfit: {'TAK' if best['overfit'] else 'NIE'}")
        lines.append("")
        lines.append(f"[-] Najgorszy: {worst['branch']}, depth={worst['depth']}")
        lines.append(f"    Acc={worst['acc'] * 100:.2f}%  F1={worst['f1'] * 100:.2f}%")
        lines.append("")

        overfit_count = sum(1 for r in self.all_results if r['overfit'])
        lines.append(f"[!] Eksperymenty z overfittingiem (>5% train-test): "
                     f"{overfit_count}/{len(self.all_results)}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  SREDNIE PER GALAZ:")
        lines.append("-" * 60)
        for branch in self.branches:
            br = [r for r in self.all_results if r['branch'] == branch]
            avg_acc = np.mean([r['acc'] for r in br]) * 100
            avg_train = np.mean([r['train_acc'] for r in br]) * 100
            avg_gap = avg_train - avg_acc
            lines.append(f"  {branch} ({len(self.branches[branch])} cech):")
            lines.append(f"    Test Acc={avg_acc:.2f}%  Train={avg_train:.2f}%  "
                         f"Gap={avg_gap:.2f}%  "
                         f"Prec={np.mean([r['prec'] for r in br]) * 100:.2f}%  "
                         f"Rec={np.mean([r['rec'] for r in br]) * 100:.2f}%  "
                         f"F1={np.mean([r['f1'] for r in br]) * 100:.2f}%")

        lines.append("")
        lines.append("-" * 60)
        lines.append("  WPLYW max_depth NA ACCURACY:")
        lines.append("-" * 60)
        for branch in self.branches:
            br = sorted([r for r in self.all_results if r['branch'] == branch],
                        key=lambda r: r['depth'])
            if len(br) >= 2:
                f_acc = br[0]['acc'] * 100
                l_acc = br[-1]['acc'] * 100
                peak = max(br, key=lambda r: r['acc'])
                trend = ("rosnie" if l_acc > f_acc else
                         "maleje" if l_acc < f_acc else "stabilna")
                lines.append(f"  {branch}: trend={trend}, "
                             f"depth {br[0]['depth']}->{br[-1]['depth']}: "
                             f"{f_acc:.1f}%->{l_acc:.1f}%, "
                             f"szczyt=depth{peak['depth']} ({peak['acc'] * 100:.2f}%)")

        branch_names = list(self.branches.keys())
        if len(branch_names) >= 3:
            lines.append("")
            lines.append("-" * 60)
            lines.append("  WPLYW REDUKCJI CECH:")
            lines.append("-" * 60)
            avg_accs = {b: np.mean([r['acc'] for r in self.all_results
                                    if r['branch'] == b]) * 100
                        for b in branch_names}
            b1, b2, b3 = branch_names
            lines.append(f"  G1 vs G2 (top 50%): {avg_accs[b1] - avg_accs[b2]:+.2f}%")
            lines.append(f"  G1 vs G3 (top 20%): {avg_accs[b1] - avg_accs[b3]:+.2f}%")
            if avg_accs[b2] > avg_accs[b1]:
                lines.append("  => Selekcja cech POPRAWILA wynik.")
            elif avg_accs[b2] < avg_accs[b1] - 2:
                lines.append("  => Selekcja cech POGORSZYLA wynik.")
            else:
                lines.append("  => Wplyw selekcji byl MINIMALNY.")

        lines.append("")
        lines.append("=" * 60)
        lines.append("  Uzupelnij wnioski w polach po prawej stronie.")
        lines.append("=" * 60)

        self.auto_obs_text.config(state=tk.NORMAL)
        self.auto_obs_text.delete('1.0', tk.END)
        self.auto_obs_text.insert(tk.END, "\n".join(lines))
        self.auto_obs_text.config(state=tk.DISABLED)
        self.notebook.select(6)

    def export_results_csv(self):
        if not self.all_results:
            messagebox.showwarning("Brak danych", "Najpierw uruchom eksperymenty!")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="wyniki_eksperymentow.csv")
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'nr', 'branch', 'depth',
                    'acc', 'prec', 'rec', 'f1',
                    'train_acc', 'overfit'])
                writer.writeheader()
                for r in self.all_results:
                    writer.writerow({
                        'nr': r['nr'],
                        'branch': r['branch'],
                        'depth': r['depth'],
                        'acc': f"{r['acc']:.4f}",
                        'prec': f"{r['prec']:.4f}",
                        'rec': f"{r['rec']:.4f}",
                        'f1': f"{r['f1']:.4f}",
                        'train_acc': f"{r['train_acc']:.4f}",
                        'overfit': 'TAK' if r['overfit'] else 'NIE',
                    })
            self.log(f"Wyniki zapisane: {path}")
            self.set_status(f"Eksport CSV: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Błąd exportu", str(e))

    def export_observations_txt(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            initialfile="wnioski_projektu.txt")
        if not path:
            return
        try:
            self.auto_obs_text.config(state=tk.NORMAL)
            auto = self.auto_obs_text.get('1.0', tk.END).strip()
            self.auto_obs_text.config(state=tk.DISABLED)

            obs = self.obs_text.get('1.0', tk.END).strip()
            wn = self.wn_text.get('1.0', tk.END).strip()
            end = self.end_text.get('1.0', tk.END).strip()

            if not any([auto, obs, wn, end]):
                messagebox.showwarning(
                    "Brak treści",
                    "Brak danych do zapisania.\n"
                    "Uruchom eksperymenty i uzupełnij pola wniosków.")
                return

            sections = [
                ("AUTOMATYCZNE SPOSTRZEZENIA", auto),
                ("OBSERWACJE", obs),
                ("WNIOSKI", wn),
                ("WNIOSKI KONCOWE", end),
            ]

            with open(path, 'w', encoding='utf-8') as f:
                for title, content in sections:
                    f.write(f"{title}\n")
                    f.write("=" * 60 + "\n")
                    f.write(content if content else "(brak treści)")
                    f.write("\n\n")

            self.log(f"Wnioski zapisane: {path}")
            self.set_status(f"Eksport TXT: {os.path.basename(path)}")
            messagebox.showinfo("Sukces", f"Plik zapisany:\n{path}")
        except Exception as e:
            messagebox.showerror("Błąd eksportu", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MLProjectGUI(root)
    root.mainloop()
