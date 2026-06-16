from theme import *
import tkinter as tk
from tkinter import ttk


# funkcje tworzące podstawowe części UI
def w98_frame(parent, **kw):  # ramka
    return tk.Frame(parent, **{**STYLES['frame'], **kw})


def w98_label(parent, text, bold=False, **kw):  # etykieta
    style = {'bg': W98['bg'], 'fg': W98['text'],
             'font': W98['font_bold'] if bold else W98['font']}
    return tk.Label(parent, text=text, **{**style, **kw})


def w98_button(parent, text, command, **kw):  # przycisk
    return tk.Button(parent, text=text, command=command,
                     **{**STYLES['button'], **kw})


def w98_entry(parent, **kw):  # pole tekstowe
    return tk.Entry(parent, **{**STYLES['entry'], **kw})


def w98_listbox(parent, **kw):  # lista
    return tk.Listbox(parent, **{**STYLES['listbox'], **kw})


def w98_scrollbar(parent, **kw):  # pasek przewijania
    return tk.Scrollbar(parent, **kw)


# funkcje tworzące bardziej skomplikowane części UI
def w98_labelframe(parent, text, **kw):  # ramka z nagłówkiem
    outer = w98_frame(parent, **kw)
    w98_label(outer, text=f" {text} ", bold=True).pack(anchor=tk.W, padx=4)
    inner = w98_frame(outer, relief=tk.GROOVE, bd=2)
    inner.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
    return outer, inner


def w98_title_bar(parent, title):  # pasek tytułu
    bar = tk.Frame(parent, bg=W98['title_bg'], height=20)
    bar.pack(fill=tk.X)
    bar.pack_propagate(False)
    tk.Label(bar, text=f"{title}", bg=W98['title_bg'], fg=W98['title_fg'],
             font=W98['font_bold'], anchor=tk.W).pack(side=tk.LEFT, fill=tk.Y)
    return bar


def w98_separator(parent):  # poziomy separator
    tk.Frame(parent, bg=W98['bg_dark'], height=1).pack(fill=tk.X, pady=2)
    tk.Frame(parent, bg='white', height=1).pack(fill=tk.X)


def w98_text_area(parent, height=5, mono=False, **kw):  # pole tekstowe z przew
    frame = w98_frame(parent, relief=tk.SUNKEN, bd=2)
    sb = w98_scrollbar(frame)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    local_params = {
        'height': height,
        'yscrollcommand': sb.set
    }
    if mono:
        local_params['font'] = W98['font_mono']
    final_opts = {**STYLES['text'], **local_params, **kw}

    txt = tk.Text(frame, **final_opts)
    txt.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    sb.config(command=txt.yview)

    return frame, txt


def w98_treeview(parent, height=10, **kw):  # tabela treeview
    frame = w98_frame(parent)
    tree = ttk.Treeview(frame, style="W98.Treeview", height=height, **kw)

    sy = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview,
                       style="W98.Vertical.TScrollbar")
    sx = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview,
                       style="W98.Horizontal.TScrollbar")

    tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)

    sy.pack(side=tk.RIGHT, fill=tk.Y)
    sx.pack(side=tk.BOTTOM, fill=tk.X)
    tree.pack(fill=tk.BOTH, expand=True)

    return frame, tree


def w98_scrolled_listbox(parent, **kw):
    frame = w98_frame(parent, relief=tk.SUNKEN, bd=2)
    sb = w98_scrollbar(frame)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    lb = w98_listbox(frame, yscrollcommand=sb.set, **kw)
    lb.pack(fill=tk.BOTH, expand=True)
    sb.config(command=lb.yview)

    return frame, lb
