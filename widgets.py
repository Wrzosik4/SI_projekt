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


def w98_entry(parent, **kw):  # pole tekstowe
    return tk.Entry(parent, **{**STYLES['entry'], **kw})


def w98_listbox(parent, **kw):  # lista
    return tk.Listbox(parent, **{**STYLES['listbox'], **kw})


def w98_scrollbar(parent, **kw):  # pasek przewijania
    return tk.Scrollbar(parent, **kw)


def w98_separator(parent):  # poziomy separator
    tk.Frame(parent, bg=W98['bg_dark'], height=1).pack(fill=tk.X, pady=2)
    tk.Frame(parent, bg='white', height=1).pack(fill=tk.X)


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


def w98_button(parent, text, command, width=None, **kw):  # przycisk
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


# ttk notebook, ale W98
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


class W98Menubar:
    def __init__(self, parent, app):
        self.app = app
        self.bar = w98_frame(parent, relief=tk.RAISED, bd=1)
        self.bar.pack(fill=tk.X)
        self._open_menu = None
        self._build()

    def _build(self):
        menus = {
            "Plik": [
                ("Wczytaj CSV...", self.app.load_data),
                ("Eksportuj wyniki CSV...", self.app.export_results_csv),
                ("Eksportuj raport TXT...", self.app.export_observations_txt),
                None,
                ("Wyjdź", self.app.root.quit),
            ],
            "Widok": [
                ("Analiza zbioru", lambda: self.app.notebook.select(0)),
                ("Konfiguracja gałęzi", lambda: self.app.notebook.select(1)),
                ("Tabela wyników", lambda: self.app.notebook.select(2)),
                ("Wykresy porównawcze", lambda: self.app.notebook.select(3)),
                ("Krzywa uczenia", lambda: self.app.notebook.select(4)),
                ("Najlepszy model", lambda: self.app.notebook.select(5)),
                ("Obserwacje i wnioski", lambda: self.app.notebook.select(6)),
                ("Instrukcja i opis", lambda: self.app.notebook.select(7)),
            ],
            "Narzędzia": [
                ("Analiza danych", self.app.analyze_data),
                ("Wybór cech", self.app.select_features),
                ("Uruchom eksperymenty...", self.app.ask_experiment_count),
                None,
                ("Wyczyść log", self.app.clear_log),
            ],
            "Pomoc": [
                ("O programie", self._show_about),
            ],
        }

        for name, items in menus.items():
            btn = tk.Button(
                self.bar, text=name, bg=W98['bg'], fg=W98['text'],
                font=W98['font'], relief=tk.FLAT, bd=1, padx=6, pady=2,
                activebackground=W98['title_bg'], activeforeground='white',
                cursor='arrow')
            btn.pack(side=tk.LEFT)
            btn.bind("<Button-1>", lambda e, b=btn,
                     i=items: self._toggle_menu(b, i))
            btn.bind("<Enter>", lambda e, b=btn, i=items: self._on_hover(b, i))

        self.app.root.bind("<Button-1>", self._close_on_click, add=True)

    def _toggle_menu(self, btn, items):
        self._close()
        self._show_dropdown(btn, items)

    def _on_hover(self, btn, items):
        if self._open_menu:
            self._close()
            self._show_dropdown(btn, items)

    def _show_dropdown(self, btn, items):
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height()

        menu = tk.Toplevel(self.app.root)
        menu.wm_overrideredirect(True)
        menu.wm_geometry(f"+{x}+{y}")
        menu.configure(bg=W98['bg'])
        menu.attributes('-topmost', True)

        outer = w98_frame(menu, bg=W98['bg_dark'], bd=1, relief=tk.SOLID)
        outer.pack()
        inner = w98_frame(outer, relief=tk.RAISED, bd=1)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        for item in items:
            if item is None:
                w98_frame(inner, bg=W98['bg_dark'],
                          height=1).pack(fill=tk.X, padx=2, pady=1)
                continue

            label, cmd = item
            row = w98_frame(inner, cursor='arrow')
            row.pack(fill=tk.X)

            lbl = w98_label(row, label, anchor=tk.W, padx=14, pady=3)
            lbl.pack(fill=tk.X)

            def bind_events(row_widget, label_widget, command):
                for widget in (row_widget, label_widget):
                    widget.bind("<Enter>", lambda e: (
                        row_widget.config(bg=W98['title_bg']),
                        label_widget.config(bg=W98['title_bg'], fg='white')
                    ))
                    widget.bind("<Leave>", lambda e: (
                        row_widget.config(bg=W98['bg']),
                        label_widget.config(bg=W98['bg'], fg=W98['text'])
                    ))
                    widget.bind("<Button-1>", lambda e: (self._close(),
                                                         command()))

            bind_events(row, lbl, cmd)
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
        w98_title_bar(win, "O programie")
        w98_label(win, "\nŚrodowisko Eksperymentów ML\n",
                  font=W98['font_title']).pack()
        w98_label(win, "DecisionTreeClassifier · scikit-learn\n"
                       "SelectKBest · f_classif\n"
                       "Obsługuje zbiory ≥1 mln rekordów").pack()

        w98_button(win, "OK", win.destroy, width=10).pack(pady=10)
