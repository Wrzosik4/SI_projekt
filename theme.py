import tkinter as tk


# paleta kolorów
W98 = {
    'bg': '#c0c0c0', 'bg_dark': '#808080', 'bg_light': '#ffffff',
    'title_bg': '#000080', 'title_fg': '#ffffff',
    'btn_bg': '#c0c0c0', 'btn_active': '#c0c0c0',
    'text': '#000000', 'disabled': '#808080', 'highlight': '#000080',
    'console_bg': '#000000', 'console_fg': '#c0c0c0',
    'select_bg': '#000080', 'select_fg':  '#ffffff',
    'font': ('Microsoft Sans Serif', 8), 'font_mono': ('Courier New', 8),
    'font_bold': ('Microsoft Sans Serif', 8, 'bold'),
    'font_title': ('Microsoft Sans Serif', 10, 'bold'),
}


# STYLE konkretnych elementów
STYLES = {
    'frame':   {'bg': W98['bg']},
    'button':  {'bg': W98['btn_bg'], 'fg': W98['text'], 'font': W98['font'],
                'relief': tk.RAISED, 'bd': 2,
                'activebackground': W98['btn_active'],
                'activeforeground': W98['text'], 'cursor': 'arrow'},
    'entry':   {'bg': W98['bg_light'], 'fg': W98['text'], 'font': W98['font'],
                'relief': tk.SUNKEN, 'bd': 2, 'insertbackground': W98['text']},
    'listbox': {'bg': W98['bg_light'], 'fg': W98['text'],
                'font': W98['font_mono'], 'relief': tk.SUNKEN, 'bd': 2,
                'selectbackground': W98['select_bg'],
                'selectforeground': W98['select_fg']},
    'text':    {'bg': W98['bg_light'], 'fg': W98['text'], 'font': W98['font'],
                'relief': tk.FLAT, 'insertbackground': W98['text']}
}


TTK_STYLES = {
    "W98.Treeview": dict(
        background=W98['bg_light'], fieldbackground=W98['bg_light'],
        foreground=W98['text'], font=W98['font'], rowheight=18),
    "W98.Treeview.Heading": dict(
        background=W98['bg'], foreground=W98['text'],
        font=W98['font_bold'], relief='raised'),
    "W98.Horizontal.TProgressbar": dict(
        troughcolor=W98['bg_light'], background=W98['title_bg'], thickness=14),
    "W98.Vertical.TScrollbar": dict(background=W98['bg']),
    "W98.Horizontal.TScrollbar": dict(background=W98['bg'])
}
