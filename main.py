from widgets import (w98_frame, w98_entry, w98_button, w98_title_bar,
                     w98_label, w98_separator, w98_labelframe, w98_text_area,
                     w98_treeview, w98_scrolled_listbox)
from theme import W98
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class W98Notebook:
    def __init__(self, parent):
        self.parent = parent
        self.tabs = []  # lista przechowująca zakładki.
        self.current = 0  # numer aktualnie wybranej zakładki.

        self.tab_bar = w98_frame(parent)  # pasek z przyciskami zakładek.
        self.tab_bar.pack(fill=tk.X)

        # ramka otaczająca zawartość
        border = w98_frame(parent, bg=W98['bg_dark'], bd=1, relief=tk.RAISED)
        border.pack(fill=tk.BOTH, expand=True)

        # główny obszar wyświetlający zawartość
        self.content = w98_frame(border, padx=1, pady=1)
        self.content.pack(fill=tk.BOTH, expand=True)

    def add(self, text):  # dodaje nową zakładkę
        idx = len(self.tabs)
        frame = w98_frame(self.content)
        btn = w98_button(
            self.tab_bar, text=text,
            command=lambda i=idx: self.select(i),
            bg=W98['bg'] if idx == 0 else W98['bg_dark'],
            relief=tk.RAISED if idx == 0 else tk.FLAT,
            padx=6, pady=2
        )
        btn.pack(side=tk.LEFT, padx=(2 if idx == 0 else 0, 0), pady=(4, 0))
        self.tabs.append((text, frame, btn))  # dodaje zakładkę do listy
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


#   Główna klasa projektu
class MLProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Środowisko Eksperymentów ML")
        self.root.geometry("1200x820")
        self.root.configure(bg=W98['bg'])
        self.root.resizable(True, True)

        try:
            self.root.iconbitmap(default='')
        except Exception:
            pass

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
                        background=W98['bg_light'],
                        foreground=W98['text'],
                        rowheight=18,
                        fieldbackground=W98['bg_light'],
                        font=W98['font'])
        style.configure("W98.Treeview.Heading",
                        background=W98['bg'],
                        foreground=W98['text'],
                        font=W98['font_bold'],
                        relief=tk.RAISED)
        style.map("W98.Treeview",
                  background=[('selected', W98['select_bg'])],
                  foreground=[('selected', W98['select_fg'])])
        style.configure("W98.Vertical.TScrollbar", background=W98['bg'])
        style.configure("W98.Horizontal.TScrollbar", background=W98['bg'])

    def create_widgets(self):
        # pasek tytułu aplikacji
        w98_title_bar(self.root, "Środowisko Eksperymentów ML — Projekt")

        # menu bar
        menubar = w98_frame(self.root, relief=tk.FLAT)
        menubar.pack(fill=tk.X)
        for item in ["Plik", "Widok", "Narzędzia", "Pomoc"]:
            w98_label(menubar, item, cursor='arrow',
                      padx=6, pady=2).pack(side=tk.LEFT)
        w98_separator(self.root)

        # główny układ
        main = w98_frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # sidebar
        self.sidebar = w98_frame(main, width=200, relief=tk.GROOVE, bd=2)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.sidebar.pack_propagate(False)

        # prawa strona
        right = w98_frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # log na dole
        log_outer, log_inner = w98_labelframe(right, "Konsola / Log")
        log_outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        # notebook
        self.notebook = W98Notebook(right)

        # tekst konsoli
        txt_frame, self.console = w98_text_area(
            log_inner, height=6, mono=True,
            bg=W98['console_bg'], fg=W98['console_fg'],
            insertbackground='white'
        )
        txt_frame.pack(fill=tk.BOTH, expand=True)

        self.build_sidebar()
        self.build_tabs()

        # Pasek statusu
        self.statusbar = w98_label(self.root, "  Gotowy. Wczytaj plik CSV," +
                                   "aby rozpocząć.", bg=W98['bg_dark'],
                                   fg='white', anchor=tk.W, relief=tk.SUNKEN,
                                   bd=1)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)

        self.log("System gotowy. Oczekuję na wczytanie danych...")

    def set_status(self, text):
        self.statusbar.config(text=f"  {text}")
        self.root.update()

    def build_sidebar(self):
        hdr = w98_frame(self.sidebar, bg=W98['title_bg'])
        hdr.pack(fill=tk.X, pady=(0, 8))
        w98_label(hdr, "ML Eksperymenty", bg=W98['title_bg'], fg='white',
                  bold=True, anchor=tk.W).pack(fill=tk.X, pady=8)

        steps = [
            ("1. Wczytaj Zbiór", self.load_data),
            ("2. Analiza Danych", self.analyze_data),
            ("3. Wybór Cech",    self.select_features),
            ("4. Uruchom Eksperymenty", self.ask_experiment_count),
        ]

        for label, cmd in steps:
            w98_button(self.sidebar, label, cmd, anchor=tk.W, padx=8, pady=4,
                       wraplength=180, justify=tk.LEFT).pack(fill=tk.X,
                                                             padx=6, pady=3)
        w98_separator(self.sidebar)

        w98_label(self.sidebar, "Drzewo decyzyjne\nDecisionTreeClassifier\n" +
                  "scikit-learn", fg=W98['disabled'], font=('Courier New', 7),
                  justify=tk.CENTER).pack(pady=(8, 4))

    def build_tabs(self):
        # ── TAB 1: Analiza zbioru
        self.tab_data = self.notebook.add("Analiza Zbioru")
        self._build_tab_data(self.tab_data)

        # ── TAB 2: Gałęzie
        self.tab_branches = self.notebook.add("Konfiguracja Gałęzi")
        self._build_tab_branches(self.tab_branches)

        # ── TAB 3: Tabela wyników
        self.tab_table = self.notebook.add("Tabela Wyników")
        self._build_tab_table(self.tab_table)

        # ── TAB 4: Wykresy
        self.tab_charts = self.notebook.add("Wykresy Porównawcze")
        self.compare_chart_frame = w98_frame(self.tab_charts, bg=W98['bg_light'])
        self.compare_chart_frame.pack(fill=tk.BOTH, expand=True)

        # ── TAB 5: Najlepszy model
        self.tab_results = self.notebook.add("Najlepszy Model")
        self._build_tab_results(self.tab_results)

        # ── TAB 6: Obserwacje
        self.tab_obs = self.notebook.add("Obserwacje i Wnioski")
        self.build_observations_tab(self.tab_obs)

    def _build_tab_data(self, parent):
        w98_title_bar(parent, "Informacje o wczytanym zbiorze danych")

        info_f = w98_frame(parent)
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
            w98_label(info_f, caption, bold=True, width=22, anchor=tk.E).grid(
                row=i, column=0, sticky=tk.E, padx=(0, 4), pady=1)

            lbl = w98_label(info_f, "—", anchor=tk.W)
            lbl.grid(row=i, column=1, sticky=tk.W, pady=1)
            setattr(self, attr, lbl)

        w98_separator(parent)
        w98_label(parent, "Statystyki opisowe cech:",
                  bold=True).pack(anchor=tk.W, padx=8, pady=(4, 2))

        stats_wrap, self.stats_tree = w98_treeview(parent, height=7)
        stats_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        _, self.class_chart_frame = (
                                    w98_labelframe(parent,
                                                   "Rozkład klas (cel" +
                                                   " klasyfikacji)")
        )
        self.class_chart_frame.master.pack(fill=tk.BOTH, expand=True,
                                           padx=8, pady=4)

    def _build_tab_branches(self, parent):
        w98_title_bar(parent, "Zestawy cech — 3 gałęzie eksperymentów")

        branches_f = w98_frame(parent)
        branches_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.listboxes = []
        titles = [
            "Gałąź 1: Wszystkie cechy",
            "Gałąź 2: Top 50% (SelectKBest)",
            "Gałąź 3: Top 20% (ścisła selekcja)",
        ]

        for t in titles:
            col = w98_frame(branches_f)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            w98_label(col, t, bold=True).pack(anchor=tk.W, pady=(0, 2))

            box_f, lb = w98_scrolled_listbox(col)
            box_f.pack(fill=tk.BOTH, expand=True)
            self.listboxes.append(lb)

        _, self.feat_chart_frame = w98_labelframe(parent, "Ważność cech — " +
                                                  "SelectKBest (f_classif " +
                                                  "score)")
        self.feat_chart_frame.master.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _build_tab_table(self, parent):
        w98_title_bar(parent, "Wyniki wszystkich eksperymentów")

        cols = ("Nr", "Gałąź", "max_depth", "Accuracy",
                "Precision", "Recall", "F1")

        tbl_f, self.results_tree = w98_treeview(parent, height=25,
                                                columns=cols, show='headings')
        tbl_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        widths = [35, 170, 75, 90, 90, 90, 90]
        for c, w in zip(cols, widths):
            self.results_tree.heading(c, text=c)
            self.results_tree.column(c, width=w, anchor=tk.CENTER)

        self.results_tree.tag_configure('best', background='#00ff00',
                                        foreground='#000080')
        self.results_tree.tag_configure('branch1', background='#ffffff')
        self.results_tree.tag_configure('branch2', background='#eeeeee')
        self.results_tree.tag_configure('branch3', background='#dde8ff')

    def _build_tab_results(self, parent):
        w98_title_bar(parent, "Najlepszy model spośród wszystkich" +
                      " eksperymentów")

        mf = w98_frame(parent)
        mf.pack(fill=tk.X, padx=8, pady=8)

        self.lbl_best_info = w98_label(mf, "Mierniki najlepszego modelu:",
                                       font=W98['font_title'])
        self.lbl_best_info.grid(row=0, column=0, columnspan=4, sticky=tk.W,
                                pady=(0, 6))

        metrics = [
            ("Accuracy: —", "lbl_acc"),
            ("Precision: —", "lbl_prec"),
            ("Recall: —", "lbl_rec"),
            ("F1-Score: —", "lbl_f1")
        ]

        for col, (text, attr) in enumerate(metrics):
            lbl = w98_label(mf, text, relief=tk.GROOVE, bd=1, padx=8, pady=4)
            lbl.grid(row=1, column=col, sticky=tk.W, padx=8, pady=2)
            setattr(self, attr, lbl)

        _, self.cm_frame = w98_labelframe(parent, "Macierz Pomyłek" +
                                          " (Confusion Matrix)")
        self.cm_frame.master.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def build_observations_tab(self, parent):
        w98_title_bar(parent, "Obserwacje i wnioski z eksperymentów")

        paned = w98_frame(parent)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = w98_frame(paned)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        w98_label(left, "Automatyczne spostrzeżenia:", bold=True).pack(anchor=tk.W, pady=(0, 2))

        auto_f, self.auto_obs_text = w98_text_area(
            left, mono=True, wrap=tk.WORD, state=tk.DISABLED,
            bg=W98['console_bg'], fg='#00ff00'
        )
        auto_f.pack(fill=tk.BOTH, expand=True)

        right = w98_frame(paned)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sections = [
            ("Obserwacje (wpisz swoje spostrzeżenia):", "obs_text"),
            ("Wnioski (na podstawie obserwacji):",      "wn_text"),
            ("Wnioski końcowe do projektu:",            "end_text"),
        ]

        for caption, attr in sections:
            w98_label(right, caption, bold=True).pack(anchor=tk.W, pady=(6, 1))

            f, txt = w98_text_area(right, height=5, wrap=tk.WORD)
            f.pack(fill=tk.BOTH, expand=True)

            setattr(self, attr, txt)

    def log(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"> {message}\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
        self.root.update()

    # ═══════════════════════════════════════════════════════════════════════
    # WCZYTYWANIE DANYCH
    # ═══════════════════════════════════════════════════════════════════════

    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files",
                                                          "*.csv")])
        if not filepath:
            return
        self.set_status(f"Wczytywanie: {filepath}")
        self.log(f"Wczytywanie: {filepath} ...")
        try:
            self.temp_raw_df = pd.read_csv(filepath)
            self.log(f"Wczytano — {self.temp_raw_df.shape[0]} wierszy," +
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

        w98_title_bar(self.popup, "Wybierz kolumny do usunięcia")

        w98_label(self.popup, "Zaznacz kolumny, które chcesz ZIGNOROWAĆ " +
                  "(usunąć).", bold=True).pack(pady=(8, 2), padx=10,
                                               anchor=tk.W)
        w98_label(self.popup, "Ostatnia niezaznaczona kolumna = kolumna " +
                  "docelowa (Target).").pack(padx=10, anchor=tk.W)

        list_frame, self.col_listbox = (
            w98_scrolled_listbox(self.popup, selectmode=tk.MULTIPLE)
        )
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.col_listbox.insert(tk.END, *self.temp_raw_df.columns)

        w98_separator(self.popup)

        bf = w98_frame(self.popup)
        bf.pack(fill=tk.X, padx=10, pady=8)

        w98_button(bf, "✔  Zatwierdź", self.apply_column_selection,
                   width=16).pack(side=tk.LEFT, padx=(0, 8))
        w98_button(bf, "✖  Anuluj", self._cancel_load,
                   width=16).pack(side=tk.LEFT)

    def _cancel_load(self):
        self.temp_raw_df = None
        self.popup.destroy()
        self.log("Anulowano wczytywanie.")
        self.set_status("Gotowy.")

    def apply_column_selection(self):
        sel = self.col_listbox.curselection()
        if sel:
            to_drop = [self.col_listbox.get(i) for i in sel]
            self.temp_raw_df.drop(columns=to_drop, inplace=True)
            self.log(f"Usunięto kolumny: {', '.join(to_drop)}")

        missing_before = self.temp_raw_df.isnull().sum().sum()
        dtypes = self.temp_raw_df.dtypes

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
                'numeric': sum(dtypes != 'object'),
                'object': sum(dtypes == 'object')
            }
            msg = f"Zbiór gotowy: {r:,} rekordów, {c-1} cech + cel '{target}'."
            self.log(msg)
            self.set_status(msg)

        self.temp_raw_df = None
        self.popup.destroy()

    # ═══════════════════════════════════════════════════════════════════════
    # ANALIZA DANYCH
    # ═══════════════════════════════════════════════════════════════════════
    def analyze_data(self):
        if self.df is None:
            return messagebox.showwarning("Uwaga",
                                          "Najpierw wczytaj dane (krok 1)!")

        m, target_col = self._meta, self._meta['target']
        class_pct = self.df[target_col].value_counts(normalize=True) * 100

        self.lbl_records.config(text=f"{m['rows']:,}")
        self.lbl_features.config(text=f"{m['features']}")
        self.lbl_types.config(
            text=(f"numeryczne: {m['numeric']}, tekstowe (wykluczone): " +
                  f"{m['object']}"))
        self.lbl_missing.config(text=f"{m['missing']:,}")
        self.lbl_target.config(text=f"'{target_col}'  ({len(class_pct)} klas)")

        balance_str = ",  ".join([f"'{k}': {v:.1f}%"
                                  for k, v in class_pct.items()])
        self.lbl_balance.config(text=balance_str)

        stats = self.df.drop(columns=[target_col]).describe().T.round(4)
        self.stats_tree.delete(*self.stats_tree.get_children())

        self.stats_tree.config(columns=['Cecha', *stats.columns],
                               show='headings')
        self.stats_tree.heading('Cecha', text='Cecha')
        self.stats_tree.column('Cecha', width=160, anchor=tk.W)

        for c in stats.columns:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=85, anchor=tk.CENTER)

        for feat, row in stats.iterrows():
            self.stats_tree.insert('', tk.END, values=[feat, *row])

        for w in self.class_chart_frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(6, 2.2), facecolor='#c0c0c0')
        ax.set(facecolor='#ffffff', title="Rozkład klas", ylabel="%")

        ax.set_facecolor('#ffffff')

        bars = ax.bar(class_pct.index.astype(str), class_pct.values,
                      edgecolor='black', linewidth=0.8,
                      color=['#000080', '#808080', '#c0c0c0',
                             '#008080', '#800000'])

        ax.bar_label(bars, fmt='%.1f%%', fontsize=7, padding=2)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.class_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

        if any(v < 30.0 for v in class_pct.values):
            self.log("OSTRZEŻENIE: Niezbalansowanie klas! Zwróć uwagę " +
                     "na Recall i Precision.")

        self.log("Analiza zakończona.")
        self.set_status("Analiza danych zakończona.")
        self.notebook.select(0)

    # ═══════════════════════════════════════════════════════════════════════
    # WYBÓR CECH
    # ═══════════════════════════════════════════════════════════════════════

    def select_features(self):
        if self.df is None:
            return messagebox.showwarning("Uwaga", "Najpierw wczytaj dane (krok 1)!")

        self.log("Selekcja cech — SelectKBest / f_classif ...")
        self.set_status("Obliczanie ważności cech...")
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        n = len(X.columns)

        selector = SelectKBest(score_func=f_classif, k='all').fit(X, y)
        scores = selector.scores_
        self.feature_scores = dict(zip(X.columns, scores))
        best_idx = np.argsort(scores)[::-1]

        self.branches = {
            'Gałąź 1': list(X.columns),
            'Gałąź 2': list(X.columns[np.sort(best_idx[:max(1, n // 2)])]),
            'Gałąź 3': list(X.columns[np.sort(best_idx[:max(1,
                                                            int(n * 0.2))])])
        }

        for i, feats in enumerate(self.branches.values()):
            self.listboxes[i].delete(0, tk.END)
            self.listboxes[i].insert(tk.END, *feats)

        counts = [f"G{i+1}: {len(feats)}" for i, feats in
                  enumerate(self.branches.values())]
        self.log(f"Wybrane cechy — {' | '.join(counts)}")

        self._draw_feature_importance()
        self.set_status("Selekcja cech zakończona — 3 gałęzie gotowe.")
        self.notebook.select(1)

    def _draw_feature_importance(self):
        for w in self.feat_chart_frame.winfo_children():
            w.destroy()

        top_feats = sorted(self.feature_scores.items(), key=lambda x: x[1],
                           reverse=True)[:20]
        if not top_feats:
            return

        names, values = zip(*top_feats)
        n = len(names)

        fig, ax = plt.subplots(figsize=(8, max(2.5, n * 0.3)),
                               facecolor='#c0c0c0')
        ax.set(
            facecolor='#ffffff',
            title="Ważność cech — SelectKBest (f_classif)", xlabel="Score")

        colors = ['#000080' if i < max(1, n // 5) else
                  '#808080' if i < n // 2 else '#c0c0c0'
                  for i in range(n)]

        ax.barh(names, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.invert_yaxis()
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.feat_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # OKIENKO LICZBY EKSPERYMENTÓW
    # ═══════════════════════════════════════════════════════════════════════

    def ask_experiment_count(self):
        if not self.branches:
            messagebox.showwarning("Uwaga",
                                   "Najpierw wykonaj wybór cech (krok 3)!")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Uruchom eksperymenty")
        popup.geometry("380x240")
        popup.configure(bg=W98['bg'])
        popup.resizable(False, False)
        popup.grab_set()

        w98_title_bar(popup, "Konfiguracja eksperymentów")

        cf = w98_frame(popup)
        cf.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        w98_label(cf, "Podaj łączną liczbę eksperymentów (min. 30).\n" +
                  "Zostaną równo podzielone na 3 gałęzie.",
                  bold=False).pack(anchor=tk.W, pady=(0, 12))

        row = w98_frame(cf)
        row.pack(fill=tk.X)
        w98_label(row, "Liczba eksperymentów:").pack(side=tk.LEFT)

        var = tk.StringVar(value="30")
        vcmd = (self.root.register(lambda v: v == "" or v.isdigit()), '%P')

        entry = w98_entry(row, textvariable=var, width=7, validate='key',
                          validatecommand=vcmd)
        entry.pack(side=tk.LEFT, padx=8)

        lbl_hint = w98_label(cf, "Na gałąź: ~10", fg=W98['disabled'])
        lbl_hint.pack(anchor=tk.W, pady=(6, 0))

        def update_hint(*_):
            try:
                lbl_hint.config(text=f"Na gałąź: ~{max(1, int(var.get())//3)}")
            except ValueError:
                lbl_hint.config(text="Na gałąź: —")

        var.trace_add("write", update_hint)

        w98_separator(cf)

        def confirm():
            raw = var.get().strip()
            if not raw or not raw.isdigit() or int(raw) < 29:
                messagebox.showwarning("Błąd",
                                       "Podaj liczbę całkowitą min 30.",
                                       parent=popup)
                return
            popup.destroy()
            self.run_experiments(int(raw))

        bf = w98_frame(cf)
        bf.pack(fill=tk.X, pady=(10, 0))
        w98_button(bf, "▶  Uruchom", confirm, width=14).pack(side=tk.LEFT,
                                                             padx=(0, 8))
        w98_button(bf, "Anuluj", popup.destroy, width=10).pack(side=tk.LEFT)

        entry.focus_set()
        entry.select_range(0, tk.END)
        popup.bind("<Return>", lambda e: confirm())
        popup.bind("<Escape>", lambda e: popup.destroy())

    def run_experiments(self, total=30):
        n_per_branch = max(1, total // 3)
        actual_total = n_per_branch * 3

        self.all_results = []
        self.results_tree.delete(*self.results_tree.get_children())
        self.log(f"Start: {actual_total} eksperymentów ({n_per_branch}/gałąź x 3)...")
        self.set_status(f"Trwa {actual_total} eksperymentów...")
        self.all_results = []
        self.results_tree.delete(*self.results_tree.get_children())

        y = self.df[self._meta['target']]

        depths = [5] if n_per_branch == 1 else np.unique(np.linspace(
            1, 30, n_per_branch, dtype=int))

        exp_nr = 1
        best_acc = -1
        best_result, best_cm = None, None

        for branch_name, features in self.branches.items():
            X = self.df[features]
            (X_train, X_test,
             y_train, y_test) = train_test_split(X, y, test_size=0.3,
                                                 random_state=42)

            for depth in depths:
                model = DecisionTreeClassifier(max_depth=int(depth),
                                               random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted',
                                       zero_division=0)
                rec  = recall_score(y_test, y_pred, average='weighted',
                                    zero_division=0)
                f1   = f1_score(y_test, y_pred, average='weighted',
                                zero_division=0)

                result = {'nr': exp_nr, 'branch': branch_name,
                          'depth': int(depth), 'acc': acc, 'prec': prec,
                          'rec': rec, 'f1': f1, 'classes': model.classes_}
                self.all_results.append(result)

                self.log(f"Exp {exp_nr:>3}/{actual_total} | {branch_name} depth={int(depth):>2} | "
                         f"Acc={acc*100:.1f}% Prec={prec*100:.1f}% Rec={rec*100:.1f}% F1={f1*100:.1f}%")
                self.set_status(f"Eksperyment {exp_nr}/{actual_total} — {branch_name}, depth={int(depth)}")

                if acc > best_acc:
                    best_acc = acc
                    best_result = result
                    best_cm = confusion_matrix(y_test, y_pred)

                exp_nr += 1

        msg = f"Najlepszy: {best_result['branch']}, depth={best_result['depth']}, Acc={best_result['acc']*100:.2f}%"

        self.log(f"=== ZAKOŃCZONO {actual_total} EKSPERYMENTÓW ===")
        self.log(msg)
        self.set_status(f"Zakończono! {msg}")

        self._fill_results_table(best_result)
        self._draw_comparison_charts()
        self._update_best_model(best_result, best_cm)
        self._generate_auto_observations()
        self.notebook.select(2)

    def _fill_results_table(self, best_result):
        self.results_tree.delete(*self.results_tree.get_children())
        tag_map = {'Gałąź 1': 'branch1', 'Gałąź 2': 'branch2',
                   'Gałąź 3': 'branch3'}
        for r in self.all_results:
            tag = 'best' if r['nr'] == best_result['nr'] else tag_map.get(r['branch'], 'branch1')
            self.results_tree.insert('', tk.END, tags=(tag,), values=(
                r['nr'], r['branch'], r['depth'],
                f"{r['acc']*100:.2f}%", f"{r['prec']*100:.2f}%",
                f"{r['rec']*100:.2f}%", f"{r['f1']*100:.2f}%",
            ))

    def _draw_comparison_charts(self):
        for w in self.compare_chart_frame.winfo_children():
            w.destroy()

        fig = plt.figure(figsize=(12, 7))
        fig.patch.set_facecolor('#c0c0c0')
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

        metrics = [('acc','Accuracy'), ('prec','Precision'), ('rec','Recall'), ('f1','F1-Score')]
        branch_colors = {'Gałąź 1': '#000080', 'Gałąź 2': '#008000', 'Gałąź 3': '#800000'}

        for idx, (mk, mn) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            ax.set_facecolor('#ffffff')
            for bn in self.branches:
                br = [r for r in self.all_results if r['branch'] == bn]
                ax.plot([r['depth'] for r in br],
                        [r[mk]*100 for r in br],
                        marker='s', markersize=4,
                        label=bn, color=branch_colors.get(bn, 'gray'),
                        linewidth=1.5)
            ax.set_title(mn, fontsize=9, fontweight='bold')
            ax.set_xlabel("max_depth", fontsize=8)
            ax.set_ylabel("[%]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.tick_params(labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#808080')

        fig.suptitle("Wpływ max_depth i zestawu cech na mierniki jakości",
                     fontsize=10, fontweight='bold')

        canvas = FigureCanvasTkAgg(fig, master=self.compare_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _update_best_model(self, best, cm):
        self.lbl_best_info.config(
            text=f"  Najlepszy model: {best['branch']}, max_depth={best['depth']}  ")
        self.lbl_acc.config(text=f"  Accuracy:  {best['acc']*100:.2f}%  ")
        self.lbl_prec.config(text=f"  Precision: {best['prec']*100:.2f}%  ")
        self.lbl_rec.config(text=f"  Recall:    {best['rec']*100:.2f}%  ")
        self.lbl_f1.config(text=f"  F1-Score:  {best['f1']*100:.2f}%  ")

        for w in self.cm_frame.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#c0c0c0')
        ax.set_facecolor('#ffffff')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=best['classes'])
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        ax.set_title("Macierz Pomyłek (Najlepszy Model)", fontsize=9)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _generate_auto_observations(self):
        if not self.all_results:
            return

        def fm(r):
            return (
                f"Acc={r['acc']*100:.2f}%  Prec={r['prec']*100:.2f}% " +
                f"Rec={r['rec']*100:.2f}%  F1={r['f1']*100:.2f}%"
            )

        best = max(self.all_results, key=lambda r: r['acc'])
        worst = min(self.all_results, key=lambda r: r['acc'])

        lines = [
            "=" * 58, "AUTOMATYCZNE SPOSTRZEŻENIA Z EKSPERYMENTÓW",
            "=" * 58, "",
            f"[+] Najlepszy: {best['branch']}, depth={best['depth']}\n" +
            f"    {fm(best)}\n",
            f"[-] Najgorszy: {worst['branch']}, depth={worst['depth']}\n" +
            f"    {fm(worst)}\n",
            "-" * 58, "  ŚREDNIE PER GAŁĄŹ:", "-" * 58
        ]

        avg_accs = {}
        for b_name, features in self.branches.items():
            br = [r for r in self.all_results if r['branch'] == b_name]
            if not br:
                continue

            avgs = {k: np.mean([r[k] for r in br]) for k in ('acc', 'prec',
                                                             'rec', 'f1')}
            avg_accs[b_name] = avgs['acc'] * 100

            lines.append(f"  {b_name} ({len(features)} cech):\n    {fm(avgs)}")

        lines.extend(["", "-" * 58, "  WPŁYW max_depth NA ACCURACY:",
                      "-" * 58])

        for b_name in self.branches:
            br = sorted([r for r in self.all_results if r['branch'] == b_name],
                        key=lambda r: r['depth'])
            if len(br) < 2:
                continue

            f_acc, l_acc = br[0]['acc'] * 100, br[-1]['acc'] * 100
            peak = max(br, key=lambda r: r['acc'])
            trend = ("rośnie" if l_acc > f_acc else
                     "maleje" if l_acc < f_acc else
                     "stabilna")

            lines.append(
                f"{b_name}: trend={trend}, depth {br[0]['depth']}" +
                f"->{br[-1]['depth']}: "
                f"{f_acc:.1f}%->{l_acc:.1f}%, szczyt=depth{peak['depth']} " +
                f"({peak['acc']*100:.2f}%)"
                )

        branch_names = list(self.branches.keys())
        if len(branch_names) >= 3:
            b1, b2, b3 = branch_names[:3]
            diff2 = avg_accs[b1] - avg_accs[b2]
            diff3 = avg_accs[b1] - avg_accs[b3]

            lines.extend([
                "", "-" * 58, "  WPŁYW REDUKCJI CECH:", "-" * 58,
                f"  {b1} vs {b2} (top 50%): {diff2:+.2f}%",
                f"  {b1} vs {b3} (top 20%): {diff3:+.2f}%"
            ])

            if avg_accs[b2] > avg_accs[b1]:
                lines.append("  => Selekcja cech POPRAWIŁA wynik.")
            elif avg_accs[b2] < avg_accs[b1] - 2:
                lines.append("  => Selekcja cech POGORSZYŁA wynik.")
            else:
                lines.append("  => Wpływ selekcji był MINIMALNY.")

        lines.extend(["", "=" * 58,
                      "  Uzupełnij wnioski w polach po prawej stronie.",
                      "=" * 58])

        self.auto_obs_text.config(state=tk.NORMAL)
        self.auto_obs_text.delete('1.0', tk.END)
        self.auto_obs_text.insert(tk.END, "\n".join(lines))
        self.auto_obs_text.config(state=tk.DISABLED)

        self.notebook.select(5)


if __name__ == "__main__":
    root = tk.Tk()
    app = MLProjectGUI(root)
    root.mainloop()
