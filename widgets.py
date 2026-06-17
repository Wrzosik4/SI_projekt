from theme import W98, STYLES
import tkinter as tk
from tkinter import ttk


# funkcje tworzące podstawowe części UI
def w98_frame(parent, **kw):  # ramka
    return tk.Frame(parent, **{**STYLES['frame'], **kw})


def w98_label(parent, text, bold=False, **kw):  # etykieta
    style = {'bg': W98['bg'], 'fg': W98['text'],
             'font': W98['font_bold'] if bold else W98['font']}
    return tk.Label(parent, text=text, **{**style, **kw})


def w98_button(parent, text, command, width=None, **kw):
    # domyślne wartości, które można nadpisać przez **kw
    bg_color = kw.pop('bg', W98['btn_bg'])
    relief_type = kw.pop('relief', tk.RAISED)

    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg_color, fg=W98['text'], font=W98['font'],
        relief=relief_type, bd=2,
        activebackground=W98['btn_active'], activeforeground=W98['text'],
        cursor='arrow',
        **({} if width is None else {'width': width}), **kw)
    return btn


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

    font = W98['font_mono'] if mono else W98['font']
    txt = tk.Text(frame, **{**STYLES['text'], 'height': height,
                            'yscrollcommand': sb.set, 'font': font, **kw})
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


class W98Notebook:
    def __init__(self, parent):
        self.parent = parent
        self.tabs = []
        self.current = 0

        self.tab_bar = w98_frame(parent)
        self.tab_bar.pack(fill=tk.X)

        self.content_border = w98_frame(parent, bg=W98['bg_dark'],
                                        bd=1, relief=tk.RAISED)
        self.content_border.pack(fill=tk.BOTH, expand=True)

        self.content = w98_frame(self.content_border)
        self.content.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    def add(self, text):
        idx = len(self.tabs)
        is_first = (idx == 0)

        frame = w98_frame(self.content)

        btn = w98_button(
            self.tab_bar, text=text,
            padx=6, pady=2,
            bg=W98['bg'] if is_first else W98['bg_dark'],
            relief=tk.RAISED if is_first else tk.FLAT,
            command=lambda i=idx: self.select(i)
        )
        btn.pack(side=tk.LEFT, padx=(2 if is_first else 0, 0), pady=(4, 0))

        self.tabs.append((text, frame, btn))

        if is_first:
            frame.pack(fill=tk.BOTH, expand=True)
        return frame

    def select(self, idx):
        if idx == self.current or not (0 <= idx < len(self.tabs)):
            return
        _, old_frame, old_btn = self.tabs[self.current]
        old_frame.pack_forget()
        old_btn.config(bg=W98['bg_dark'], relief=tk.FLAT)

        self.current = idx

        _, new_frame, new_btn = self.tabs[idx]
        new_frame.pack(fill=tk.BOTH, expand=True)
        new_btn.config(bg=W98['bg'], relief=tk.RAISED)
