from theme import W98, TTK_STYLES
from widgets import (
    w98_label, w98_button, w98_entry, w98_listbox, w98_scrollbar,
    w98_labelframe, w98_title_bar, w98_separator, w98_frame,
    W98Notebook, W98Menubar, w98_text_area, w98_treeview,
    w98_scrolled_listbox
    )

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ═══════════════════════════════════════════════════════════════════════
# GŁÓWNA KLASA APLIKACJI
# ═══════════════════════════════════════════════════════════════════════
class MLProjectGUI:
    """
    Główna klasa aplikacji graficznej do eksperymentów ML.

    Zarządza całym interfejsem użytkownika (Tkinter) oraz logiką:
    wczytywania danych, analizy, selekcji cech, uruchamiania
    eksperymentów z DecisionTreeClassifier i eksportu wyników.

    Atrybuty:
        root (tk.Tk): Główne okno aplikacji.
        df (pd.DataFrame): Wczytany i przetworzony zbiór danych.
        temp_raw_df (pd.DataFrame): Tymczasowy surowy DataFrame przed filtrowaniem kolumn.
        branches (dict): Słownik {nazwa_gałęzi: lista_cech} dla 3 gałęzi eksperymentów.
        all_results (list): Lista słowników z wynikami wszystkich eksperymentów.
        feature_scores (dict): Słownik {nazwa_cechy: score} z wynikami SelectKBest.
        _meta (dict): Metadane wczytanego zbioru (rekordy, cechy, target, braki).
        last_total_exp (int|str): Liczba wykonanych eksperymentów (do raportu).
        last_criterion (str): Użyte kryterium drzewa (do raportu).
        last_test_size (float|str): Użyty rozmiar zbioru testowego (do raportu).
    """

    def __init__(self, root):
        """
        Inicjalizuje główne okno aplikacji i wszystkie zmienne stanu.

        Kroki:
            1. Konfiguruje okno Tk (tytuł, rozmiar, kolor tła).
            2. Inicjalizuje zmienne stanu (df, branches, results itp.).
            3. Aplikuje style TTK (motyw Windows 98).
            4. Buduje wszystkie widgety interfejsu.

        Args:
            root (tk.Tk): Główne okno Tkinter przekazane z bloku __main__.
        """
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

        self.last_total_exp = "Brak"
        self.last_criterion = "Brak"
        self.last_test_size = "Brak"

        self._apply_ttk_style()
        self.create_widgets()

    def _apply_ttk_style(self):
        """
        Aplikuje motyw wizualny Windows 98 do wszystkich widgetów TTK.

        Kroki:
            1. Ustawia bazowy motyw TTK na 'default'.
            2. Iteruje po słowniku TTK_STYLES i konfiguruje każdy styl.
            3. Dodaje mapowanie kolorów zaznaczenia dla Treeview (select_bg/fg).
        """
        s = ttk.Style()
        s.theme_use('default')

        for name, kw in TTK_STYLES.items():
            s.configure(name, **kw)

        # mapowanie interakcji
        s.map("W98.Treeview",
              background=[('selected', W98['select_bg'])],
              foreground=[('selected', W98['select_fg'])])

    def create_widgets(self):
        """
        Buduje główny układ interfejsu użytkownika.

        Kroki:
            1. Tworzy pasek tytułu i menu.
            2. Buduje główny kontener (main frame) z podziałem na sidebar i obszar prawy.
            3. Tworzy notebook (zakładki) w obszarze prawym.
            4. Dodaje pasek postępu i etykietę postępu.
            5. Dodaje konsolę/log na dole prawego obszaru.
            6. Wywołuje build_sidebar() i build_tabs().
            7. Dodaje pasek statusu na samym dole okna.
        """
        w98_title_bar(self.root, "Środowisko Eksperymentów ML — Projekt")
        self.menubar = W98Menubar(self.root, self)
        w98_separator(self.root)

        main = w98_frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        self.sidebar = w98_frame(main, bg=W98['bg'], width=200, relief=tk.GROOVE, bd=2)
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
        self.progress_lbl = tk.Label(prog_f, text="", bg=W98['bg'], fg=W98['text'], font=W98['font'])
        self.progress_lbl.pack(anchor=tk.W)

        log_outer, log_inner = w98_labelframe(right, "Konsola / Log")
        log_outer.pack(fill=tk.X, pady=(2, 0))
        self.console = tk.Text(
            log_inner, height=5, bg=W98['console_bg'], fg=W98['console_fg'],
            font=W98['font_mono'], relief=tk.SUNKEN, bd=1,
            insertbackground='white', state=tk.NORMAL)
        csb = w98_scrollbar(log_inner, command=self.console.yview)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.configure(yscrollcommand=csb.set)
        self.console.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.build_sidebar()
        self.build_tabs()

        self.statusbar = tk.Label(self.root, text="  Gotowy. Wczytaj plik CSV aby rozpocząć.",
                                  bg=W98['bg_dark'], fg='white', font=W98['font'], 
                                  anchor=tk.W, relief=tk.SUNKEN, bd=1)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
        self.log("System gotowy. Oczekuję na wczytanie danych...")

    def set_status(self, text):
        """
        Aktualizuje tekst paska statusu na dole okna i odświeża UI.

        Args:
            text (str): Tekst do wyświetlenia w pasku statusu.
        """
        self.statusbar.config(text=f"  {text}")
        self.root.update()

    def set_progress(self, value, label=""):
        """
        Ustawia wartość paska postępu i etykietę opisową.

        Args:
            value (float): Wartość postępu w zakresie 0–100.
            label (str): Opcjonalny tekst wyświetlany obok paska (domyślnie pusty).
        """
        self.progress_var.set(value)
        self.progress_lbl.config(text=label)
        self.root.update()

    def build_sidebar(self):
        """
        Buduje lewy panel boczny (sidebar) z przyciskami kroków i eksportu.

        Kroki:
            1. Tworzy nagłówek sidebaru z tytułem.
            2. Dodaje 4 przyciski kroków: Wczytaj, Analiza, Wybór cech, Eksperymenty.
            3. Dodaje separator i sekcję eksportu (CSV, TXT).
            4. Dodaje stopkę z informacją o algorytmie.
        """
        hdr = tk.Frame(self.sidebar, bg=W98['title_bg'])
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="  📁 ML Eksperymenty", bg=W98['title_bg'], fg='white',
                 font=W98['font_bold'], pady=8, anchor=tk.W).pack(fill=tk.X)

        tk.Frame(self.sidebar, bg=W98['bg'], height=8).pack()

        steps = [
            ("1. Wczytaj Zbiór", self.load_data),
            ("2. Analiza Danych", self.analyze_data),
            ("3. Wybór Cech", self.select_features),
            ("4. Uruchom\n       Eksperymenty", self.ask_experiment_count),
        ]
        for label, cmd in steps:
            btn = w98_button(self.sidebar, label, cmd)
            btn.config(anchor=tk.W, padx=8, pady=4, wraplength=180, justify=tk.LEFT)
            btn.pack(fill=tk.X, padx=6, pady=3)

        tk.Frame(self.sidebar, bg=W98['bg_dark'], height=1).pack(fill=tk.X, padx=4, pady=8)
        tk.Frame(self.sidebar, bg='white', height=1).pack(fill=tk.X, padx=4)

        w98_label(self.sidebar, "Eksport:", bold=True).pack(anchor=tk.W, padx=6, pady=(4, 0))
        w98_button(self.sidebar, "Wyniki → CSV", self.export_results_csv).pack(fill=tk.X, padx=6, pady=2)
        w98_button(self.sidebar, "Raport → TXT", self.export_observations_txt).pack(fill=tk.X, padx=6, pady=2)

        tk.Frame(self.sidebar, bg=W98['bg'], height=8).pack()
        tk.Label(self.sidebar, text="DecisionTreeClassifier\nscikit-learn",
                 bg=W98['bg'], fg=W98['disabled'], font=('Courier New', 7), justify=tk.CENTER).pack(pady=4)

    def build_tabs(self):
        """
        Tworzy i rejestruje wszystkie zakładki notebooka.

        Zakładki (w kolejności):
            0. Analiza Zbioru       — _build_tab_data()
            1. Konfiguracja Gałęzi  — _build_tab_branches()
            2. Tabela Wyników       — _build_tab_table()
            3. Wykresy Porównawcze  — pusty frame (wypełniany przez _draw_comparison_charts)
            4. Krzywa Uczenia       — pusty frame (wypełniany przez _draw_learning_curve)
            5. Najlepszy Model      — _build_tab_results()
            6. Obserwacje i Wnioski — build_observations_tab()
            7. Instrukcja           — _build_tab_instruction()
        """
        self._build_tab_data(self.notebook.add("Analiza Zbioru"))
        self._build_tab_branches(self.notebook.add("Konfiguracja Gałęzi"))
        self._build_tab_table(self.notebook.add("Tabela Wyników"))
        
        t4 = self.notebook.add("Wykresy Porównawcze")
        self.compare_chart_frame = tk.Frame(t4, bg=W98['bg_light'])
        self.compare_chart_frame.pack(fill=tk.BOTH, expand=True)

        t5 = self.notebook.add("Krzywa Uczenia")
        self.learning_curve_frame = tk.Frame(t5, bg=W98['bg_light'])
        self.learning_curve_frame.pack(fill=tk.BOTH, expand=True)

        self._build_tab_results(self.notebook.add("Najlepszy Model"))
        self.build_observations_tab(self.notebook.add("Obserwacje i Wnioski"))
        self._build_tab_instruction(self.notebook.add("Instrukcja"))

    def _build_tab_data(self, parent):
        """
        Buduje zakładkę 'Analiza Zbioru' z informacjami o danych i statystykami.

        Kroki:
            1. Tworzy siatkę etykiet z metadanymi (rekordy, cechy, typy, braki, klasy, target).
            2. Dodaje Treeview ze statystykami opisowymi cech (mean, std, min, percentyle, max).
            3. Rezerwuje ramkę na wykres słupkowy rozkładu klas (self.class_chart_frame).

        Args:
            parent (tk.Frame): Ramka zakładki przekazana przez notebook.add().
        """
        w98_title_bar(parent, "Informacje o wczytanym zbiorze danych")
        info_f = tk.Frame(parent, bg=W98['bg'])
        info_f.pack(fill=tk.X, padx=8, pady=6)

        labels_def = [
            ("Liczba rekordów:", "lbl_records"), ("Liczba cech:", "lbl_features"),
            ("Typy kolumn:", "lbl_types"), ("Brakujące wartości:", "lbl_missing"),
            ("Zbalansowanie klas:", "lbl_balance"), ("Kolumna docelowa:", "lbl_target"),
        ]
        for i, (caption, attr) in enumerate(labels_def):
            tk.Label(info_f, text=caption, bg=W98['bg'], fg=W98['text'],
                     font=W98['font_bold'], width=22, anchor=tk.E).grid(row=i, column=0, sticky=tk.E, padx=(0, 4), pady=1)
            lbl = tk.Label(info_f, text="—", bg=W98['bg'], fg=W98['text'], font=W98['font'], anchor=tk.W)
            lbl.grid(row=i, column=1, sticky=tk.W, pady=1)
            setattr(self, attr, lbl)

        w98_separator(parent)
        w98_label(parent, "Statystyki opisowe cech:", bold=True).pack(anchor=tk.W, padx=8, pady=(4, 2))

        stats_wrap = tk.Frame(parent, bg=W98['bg'])
        stats_wrap.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        self.stats_tree = ttk.Treeview(stats_wrap, style="W98.Treeview", height=7)
        sy = ttk.Scrollbar(stats_wrap, orient=tk.VERTICAL, command=self.stats_tree.yview, style="W98.Vertical.TScrollbar")
        sx = ttk.Scrollbar(stats_wrap, orient=tk.HORIZONTAL, command=self.stats_tree.xview, style="W98.Horizontal.TScrollbar")
        self.stats_tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side=tk.RIGHT, fill=tk.Y)
        sx.pack(side=tk.BOTTOM, fill=tk.X)
        self.stats_tree.pack(fill=tk.BOTH, expand=True)

        _, self.class_chart_frame = w98_labelframe(parent, "Rozkład klas (cel klasyfikacji)")
        self.class_chart_frame.master.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _build_tab_branches(self, parent):
        """
        Buduje zakładkę 'Konfiguracja Gałęzi' z listboxami cech i wykresem ważności.

        Kroki:
            1. Tworzy 3 kolumny obok siebie, każda z tytułem i listboxem (self.listboxes[0..2]).
            2. Rezerwuje ramkę na poziomy wykres słupkowy ważności cech (self.feat_chart_frame).

        Args:
            parent (tk.Frame): Ramka zakładki przekazana przez notebook.add().
        """
        w98_title_bar(parent, "Zestawy cech — 3 gałęzie eksperymentów")
        branches_f = tk.Frame(parent, bg=W98['bg'])
        branches_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.listboxes = []
        titles = ["Gałąź 1: Wszystkie cechy", "Gałąź 2: Top 50% (SelectKBest)", "Gałąź 3: Top 20% (ścisła selekcja)"]
        for t in titles:
            col = tk.Frame(branches_f, bg=W98['bg'])
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            w98_label(col, t, bold=True).pack(anchor=tk.W, pady=(0, 2))
            box_f, lb = w98_scrolled_listbox(col)
            box_f.pack(fill=tk.BOTH, expand=True)
            self.listboxes.append(lb)

        _, self.feat_chart_frame = w98_labelframe(parent, "Ważność cech — SelectKBest (f_classif score)")
        self.feat_chart_frame.master.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _build_tab_table(self, parent):
        """
        Buduje zakładkę 'Tabela Wyników' z Treeview wyników eksperymentów.

        Kroki:
            1. Definiuje kolumny tabeli (Nr, Gałąź, max_depth, metryki, Overfit).
            2. Tworzy Treeview z paskami przewijania (self.results_tree).
            3. Ustawia szerokości kolumn i konfiguruje tagi kolorystyczne:
               - 'best'    → zielone tło (najlepszy wynik)
               - 'overfit' → czerwonawe tło
               - 'branch1/2/3' → naprzemienne szarości/niebieski

        Args:
            parent (tk.Frame): Ramka zakładki przekazana przez notebook.add().
        """
        w98_title_bar(parent, "Wyniki wszystkich eksperymentów")
        tbl_f = tk.Frame(parent, bg=W98['bg'])
        tbl_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        cols = ("Nr", "Gałąź", "max_depth", "Accuracy", "Precision", "Recall", "F1", "Train Acc", "Overfit?")
        tree_frame, self.results_tree = w98_treeview(tbl_f, height=25, columns=cols, show='headings')
        tree_frame.pack(fill=tk.BOTH, expand=True)
        widths = [35, 150, 75, 85, 85, 85, 85, 85, 75]
        for c, w in zip(cols, widths):
            self.results_tree.heading(c, text=c)
            self.results_tree.column(c, width=w, anchor=tk.CENTER)

        self.results_tree.tag_configure('best', background='#00ff00', foreground='#000080')
        self.results_tree.tag_configure('overfit', background='#ffdddd')
        self.results_tree.tag_configure('branch1', background='#ffffff')
        self.results_tree.tag_configure('branch2', background='#eeeeee')
        self.results_tree.tag_configure('branch3', background='#dde8ff')

    def _build_tab_results(self, parent):
        """
        Buduje zakładkę 'Najlepszy Model' z etykietami metryk i miejscem na wykresy.

        Kroki:
            1. Tworzy wiersz etykiet na metryki (Accuracy, Precision, Recall, F1, CV).
            2. Rezerwuje ramkę self.cm_frame na macierz pomyłek (lewa strona).
            3. Rezerwuje ramkę self.tree_frame na wizualizację drzewa (prawa strona).

        Args:
            parent (tk.Frame): Ramka zakładki przekazana przez notebook.add().
        """
        w98_title_bar(parent, "Najlepszy model spośród wszystkich eksperymentów")
        mf = tk.Frame(parent, bg=W98['bg'])
        mf.pack(fill=tk.X, padx=8, pady=8)

        self.lbl_best_info = tk.Label(mf, text="Mierniki najlepszego modelu:", bg=W98['bg'], fg=W98['text'], font=W98['font_title'])
        self.lbl_best_info.grid(row=0, column=0, columnspan=5, sticky=tk.W, pady=(0, 6))

        for col, (attr, txt) in enumerate([
            ('lbl_acc', "Accuracy: —"), ('lbl_prec', "Precision: —"),
            ('lbl_rec', "Recall: —"), ('lbl_f1', "F1-Score: —"), ('lbl_cv', "CV Score: —"),
        ]):
            lbl = tk.Label(mf, text=txt, bg=W98['bg'], fg=W98['text'], font=W98['font'], relief=tk.GROOVE, bd=1, padx=8, pady=4)
            lbl.grid(row=1, column=col, sticky=tk.W, padx=6, pady=2)
            setattr(self, attr, lbl)

        split = tk.Frame(parent, bg=W98['bg'])
        split.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        _, self.cm_frame = w98_labelframe(split, "Macierz Pomyłek (Confusion Matrix)")
        self.cm_frame.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        _, self.tree_frame = w98_labelframe(split, "Wizualizacja Drzewa Decyzyjnego (głębokość ≤3)")
        self.tree_frame.master.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def build_observations_tab(self, parent):
        """
        Buduje zakładkę 'Obserwacje i Wnioski' z polem auto-raportu i polami tekstowymi.

        Kroki:
            1. Lewa strona: pole tekstowe tylko do odczytu (self.auto_obs_text)
               na automatycznie generowane spostrzeżenia (zielony tekst na czarnym tle).
            2. Prawa strona: 3 edytowalne pola tekstowe:
               - self.obs_text  — obserwacje użytkownika
               - self.wn_text   — wnioski na podstawie obserwacji
               - self.end_text  — wnioski końcowe do projektu

        Args:
            parent (tk.Frame): Ramka zakładki przekazana przez notebook.add().
        """
        w98_title_bar(parent, "Obserwacje i wnioski z eksperymentów")
        paned = tk.Frame(parent, bg=W98['bg'])
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.Frame(paned, bg=W98['bg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        w98_label(left, "Automatyczne spostrzeżenia:", bold=True).pack(anchor=tk.W, pady=(0, 2))
        auto_f = tk.Frame(left, bg=W98['console_bg'], relief=tk.SUNKEN, bd=2)
        auto_f.pack(fill=tk.BOTH, expand=True)
        asb = w98_scrollbar(auto_f)
        asb.pack(side=tk.RIGHT, fill=tk.Y)
        self.auto_obs_text = tk.Text(auto_f, bg=W98['console_bg'], fg='#00ff00', font=W98['font_mono'], 
                                     wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, yscrollcommand=asb.set)
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
            f, txt = w98_text_area(right, height=5, bg=W98['bg_light'], wrap=tk.WORD)
            f.pack(fill=tk.BOTH, expand=True)
            setattr(self, attr, txt)

    def _build_tab_instruction(self, parent):
        """
        Buduje zakładkę 'Instrukcja' z treścią wczytaną z pliku instrukcja.txt.

        Kroki:
            1. Próbuje otworzyć i wczytać plik 'instrukcja.txt' (UTF-8).
            2. W razie braku pliku lub błędu — wstawia komunikat o błędzie.
            3. Wstawia treść do pola tekstowego tylko do odczytu (czcionka mono).

        Args:
            parent (tk.Frame): Ramka zakładki przekazana przez notebook.add().
        """
        w98_title_bar(parent, "Instrukcja obsługi środowiska i interpretacji wyników")
        f, txt = w98_text_area(parent, mono=True, bg=W98['bg_light'], wrap=tk.WORD)
        f.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        try:
            with open("instrukcja.txt", "r", encoding="utf-8") as file:
                instrukcja_tresc = file.read()
        except FileNotFoundError:
            instrukcja_tresc = "BŁĄD: Nie znaleziono pliku 'instrukcja.txt'.\nUpewnij się, że plik znajduje się w tym samym folderze co aplikacja."
        except Exception as e:
            instrukcja_tresc = f"BŁĄD: Nie można wczytać instrukcji.\nSzczegóły: {str(e)}"

        txt.insert(tk.END, instrukcja_tresc)
        txt.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────────────────────────────────
    def log(self, message):
        """
        Dopisuje wiadomość do konsoli/logu i przewija widok na koniec.

        Args:
            message (str): Tekst do wyświetlenia w konsoli (poprzedzony znakiem '>').
        """
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"> {message}\n")
        self.console.see(tk.END)
        self.root.update()

    def clear_log(self):
        """Czyści całą zawartość konsoli/logu."""
        self.console.config(state=tk.NORMAL)
        self.console.delete('1.0', tk.END)

    # ─────────────────────────────────────────────────────────────────────
    # WCZYTYWANIE DANYCH
    # ─────────────────────────────────────────────────────────────────────
    def load_data(self):
        """
        Otwiera dialog wyboru pliku CSV i wczytuje go do temp_raw_df.

        Kroki:
            1. Wyświetla dialog filedialog do wyboru pliku .csv.
            2. Wczytuje plik przez pd.read_csv() do self.temp_raw_df.
            3. Loguje liczbę wierszy i kolumn.
            4. Wywołuje show_column_selector() do wyboru kolumn.
            5. W razie błędu — pokazuje messagebox i loguje wyjątek.
        """
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        self.set_status(f"Wczytywanie: {filepath}")
        self.log(f"Wczytywanie: {filepath} ...")
        try:
            self.temp_raw_df = pd.read_csv(filepath)
            self.log(f"Wczytano — {self.temp_raw_df.shape[0]} wierszy, {self.temp_raw_df.shape[1]} kolumn.")
            self.show_column_selector()
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            self.log(f"Błąd: {e}")

    def show_column_selector(self):
        """
        Wyświetla modalne okno popup do wyboru kolumn do usunięcia ze zbioru.

        Kroki:
            1. Tworzy okno Toplevel z grab_set() (blokuje główne okno).
            2. Wypełnia listbox wszystkimi kolumnami z temp_raw_df.
            3. Użytkownik zaznacza kolumny do ZIGNOROWANIA (multi-select).
            4. Przyciski: Zatwierdź → apply_column_selection(), Anuluj → _cancel_load().

        Uwaga:
            Ostatnia niezaznaczona kolumna staje się kolumną docelową (target).
        """
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Zarządzanie Kolumnami")
        self.popup.geometry("460x500")
        self.popup.configure(bg=W98['bg'])
        self.popup.grab_set()
        self.popup.resizable(False, False)

        w98_title_bar(self.popup, "Wybierz kolumny do usunięcia")
        w98_label(self.popup, "Zaznacz kolumny, które chcesz ZIGNOROWAĆ (usunąć).", bold=True).pack(pady=(8, 2), padx=10, anchor=tk.W)
        w98_label(self.popup, "Ostatnia niezaznaczona kolumna = kolumna docelowa (Target).").pack(padx=10, anchor=tk.W)

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
        w98_button(bf, "✔  Zatwierdź", self.apply_column_selection, width=16).pack(side=tk.LEFT, padx=(0, 8))
        w98_button(bf, "✖  Anuluj", self._cancel_load, width=16).pack(side=tk.LEFT)

    def _cancel_load(self):
        """
        Anuluje wczytywanie danych — czyści temp_raw_df i zamyka popup.
        """
        self.temp_raw_df = None
        self.popup.destroy()
        self.log("Anulowano wczytywanie.")
        self.set_status("Gotowy.")

    def apply_column_selection(self):
        """
        Zatwierdza wybór kolumn, przetwarza DataFrame i zapisuje do self.df.

        Kroki:
            1. Usuwa zaznaczone kolumny z temp_raw_df.
            2. Zapamiętuje statystyki przed filtrowaniem (braki, typy).
            3. Filtruje tylko kolumny numeryczne i usuwa wiersze z NaN.
            4. Próbkuje do 1.2 mln rekordów jeśli zbiór jest zbyt duży (ochrona RAM).
            5. Waliduje minimalną liczbę kolumn (min. 2: cechy + target).
            6. Zapisuje metadane do self._meta i zamyka popup.
        """
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
            msg = f"Zbiór gotowy: {r:,} rekordów, {c - 1} cech + cel '{target}'."
            self.log(msg)
            self.set_status(msg)

        self.temp_raw_df = None
        self.popup.destroy()

    # ─────────────────────────────────────────────────────────────────────
    # ANALIZA DANYCH I WYBÓR CECH
    # ─────────────────────────────────────────────────────────────────────
    def analyze_data(self):
        """
        Wypełnia zakładkę 'Analiza Zbioru' statystykami i wykresem rozkładu klas.

        Kroki:
            1. Sprawdza czy dane są wczytane.
            2. Oblicza rozkład klas (value_counts + procentowy udział).
            3. Aktualizuje etykiety metadanych (rekordy, cechy, typy, braki, target, balans).
            4. Wypełnia Treeview statystykami opisowymi (describe().T).
            5. Rysuje wykres słupkowy rozkładu klas z etykietami procentowymi.
            6. Loguje ostrzeżenie jeśli jakakolwiek klasa ma poniżej 30% udziału.
        """
        if self.df is None: return messagebox.showwarning("Uwaga", "Najpierw wczytaj dane (krok 1)!")
        m, target_col = self._meta, self._meta['target']
        class_counts = self.df[target_col].value_counts()
        class_pct = class_counts / len(self.df) * 100

        self.lbl_records.config(text=f"{m['rows']:,}")
        self.lbl_features.config(text=f"{m['features']}")
        self.lbl_types.config(text=f"numeryczne: {m['numeric']}, tekstowe (wykluczone): {m['object']}")
        self.lbl_missing.config(text=f"{m['missing']:,}")
        self.lbl_target.config(text=f"'{target_col}'  ({len(class_counts)} klas)")
        self.lbl_balance.config(text=",  ".join([f"'{k}': {v:.1f}%" for k, v in class_pct.items()]))

        stats = self.df.iloc[:, :-1].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(4)
        self.stats_tree.delete(*self.stats_tree.get_children())
        self.stats_tree['columns'] = ['Cecha'] + list(stats.columns)
        self.stats_tree['show'] = 'headings'
        self.stats_tree.heading('Cecha', text='Cecha')
        self.stats_tree.column('Cecha', width=160, anchor=tk.W)
        for c in stats.columns:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=85, anchor=tk.CENTER)
        for feat, row in stats.iterrows():
            self.stats_tree.insert('', tk.END, values=[feat] + list(row))

        for w in self.class_chart_frame.winfo_children(): w.destroy()
        fig, ax = plt.subplots(figsize=(6, 2.2))
        fig.patch.set_facecolor('#c0c0c0')
        ax.set_facecolor('#ffffff')
        colors = ['#000080', '#808080', '#c0c0c0', '#008080', '#800000']
        bars = ax.bar([str(k) for k in class_counts.index], class_pct.values, color=colors[:len(class_counts)], edgecolor='black', linewidth=0.8)
        ax.set_title("Rozkład klas", fontsize=9)
        ax.set_ylabel("%", fontsize=8)
        for bar, pct in zip(bars, class_pct.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.class_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

        if any(v < 30.0 for v in class_pct.values):
            self.log("OSTRZEŻENIE: Niezbalansowanie klas! Zwróć uwagę na Recall i Precision.")
        self.log("Analiza zakończona.")
        self.set_status("Analiza danych zakończona.")
        self.notebook.select(0)

    def select_features(self):
        """
        Wykonuje selekcję cech metodą SelectKBest (f_classif) i tworzy 3 gałęzie.

        Kroki:
            1. Sprawdza czy dane są wczytane.
            2. Uruchamia SelectKBest z f_classif na wszystkich cechach.
            3. Zapisuje wyniki do self.feature_scores.
            4. Tworzy 3 gałęzie w self.branches:
               - Gałąź 1: wszystkie cechy
               - Gałąź 2: top 50% cech wg score (posortowane wg indeksu oryginalnego)
               - Gałąź 3: top 20% cech wg score (minimum 1 cecha)
            5. Wypełnia listboxy w zakładce Konfiguracja Gałęzi.
            6. Wywołuje _draw_feature_importance() i przełącza na zakładkę 1.
        """
        if self.df is None: return messagebox.showwarning("Uwaga", "Najpierw wczytaj dane (krok 1)!")
        self.log("Selekcja cech — SelectKBest / f_classif ...")
        self.set_status("Obliczanie ważności cech...")

        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]
        n = len(X.columns)

        selector = SelectKBest(score_func=f_classif, k='all').fit(X, y)
        scores = selector.scores_
        self.feature_scores = dict(zip(X.columns, scores))

        self.branches['Gałąź 1'] = list(X.columns)
        self.branches['Gałąź 2'] = list(X.columns[np.sort(np.argsort(scores)[::-1][:max(1, n // 2)])])
        self.branches['Gałąź 3'] = list(X.columns[np.sort(np.argsort(scores)[::-1][:max(1, int(n * 0.2))])])

        for i, (_, feats) in enumerate(self.branches.items()):
            self.listboxes[i].delete(0, tk.END)
            for f in feats: self.listboxes[i].insert(tk.END, f)

        self.log(f"Wybrane cechy — Gałąź 1: {len(self.branches['Gałąź 1'])} | Gałąź 2: {len(self.branches['Gałąź 2'])} | Gałąź 3: {len(self.branches['Gałąź 3'])}")
        self._draw_feature_importance()
        self.set_status("Selekcja cech zakończona — 3 gałęzie gotowe.")
        self.notebook.select(1)

    def _draw_feature_importance(self):
        """
        Rysuje poziomy wykres słupkowy ważności cech (top 20) w zakładce Gałęzie.

        Kroki:
            1. Sortuje self.feature_scores malejąco i bierze top 20.
            2. Koloruje słupki: top 20% — granatowe, top 50% — szare, reszta — jasnoszare.
            3. Osadza wykres w self.feat_chart_frame przez FigureCanvasTkAgg.
        """
        for w in self.feat_chart_frame.winfo_children(): w.destroy()
        sorted_feats = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        names, values = [f[0] for f in sorted_feats], [f[1] for f in sorted_feats]

        fig, ax = plt.subplots(figsize=(8, max(2.5, len(names) * 0.3)))
        fig.patch.set_facecolor('#c0c0c0')
        ax.set_facecolor('#ffffff')
        colors = ['#000080' if i < max(1, len(names) // 5) else '#808080' if i < len(names) // 2 else '#c0c0c0' for i in range(len(names))]
        ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
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
        """
        Wyświetla modalne okno konfiguracji eksperymentów przed ich uruchomieniem.

        Kroki:
            1. Sprawdza czy gałęzie są zdefiniowane (krok 3 musi być wykonany).
            2. Tworzy okno Toplevel z 4 parametrami do ustawienia:
               - Liczba eksperymentów (min. 30, tylko cyfry)
               - Kryterium drzewa: 'gini' lub 'entropy'
               - Rozmiar zbioru testowego: 0.2 / 0.3 / 0.4
               - Checkbox: czy wykonać 5-fold cross-validation
            3. Etykieta podpowiedzi aktualizuje się dynamicznie (liczba exp. na gałąź).
            4. Przycisk Uruchom waliduje dane i wywołuje run_experiments().
            5. Obsługuje klawisze Enter (potwierdź) i Escape (anuluj).
        """
        if not self.branches: return messagebox.showwarning("Uwaga", "Najpierw wykonaj wybór cech (krok 3)!")

        popup = tk.Toplevel(self.root)
        popup.title("Uruchom eksperymenty")
        popup.geometry("400x380")
        popup.configure(bg=W98['bg'])
        popup.resizable(False, False)
        popup.grab_set()

        w98_title_bar(popup, "Konfiguracja eksperymentów")
        cf = tk.Frame(popup, bg=W98['bg'])
        cf.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        w98_label(cf, "Skonfiguruj parametry startowe algorytmu.\nLiczba eksperymentów zostanie podzielona na 3 gałęzie.", bold=True).pack(anchor=tk.W, pady=(0, 8))

        # 1. PARAMETR: Liczba eksperymentów
        row1 = tk.Frame(cf, bg=W98['bg'])
        row1.pack(fill=tk.X, pady=(4, 0))
        w98_label(row1, "Liczba eksperymentów (min. 30):").pack(side=tk.LEFT)
        var_exp = tk.StringVar(value="30")
        vcmd = (self.root.register(lambda v: v == "" or v.isdigit()), '%P')
        entry = w98_entry(row1, textvariable=var_exp, width=7, validate='key', validatecommand=vcmd)
        entry.pack(side=tk.LEFT, padx=8)
        lbl_hint = tk.Label(cf, text="Na gałąź: ~10", bg=W98['bg'], fg=W98['disabled'], font=W98['font'])
        lbl_hint.pack(anchor=tk.W, pady=(2, 8))

        # 2. PARAMETR: Kryterium podziału
        row2 = tk.Frame(cf, bg=W98['bg'])
        row2.pack(fill=tk.X, pady=4)
        w98_label(row2, "Kryterium drzewa:").pack(side=tk.LEFT)
        var_crit = tk.StringVar(value="gini")
        cb_crit = ttk.Combobox(row2, textvariable=var_crit, values=["gini", "entropy"], state="readonly", width=10)
        cb_crit.pack(side=tk.LEFT, padx=(22, 8))

        # 3. PARAMETR: Zbiór testowy
        row3 = tk.Frame(cf, bg=W98['bg'])
        row3.pack(fill=tk.X, pady=4)
        w98_label(row3, "Rozmiar testowy:").pack(side=tk.LEFT)
        var_test = tk.StringVar(value="0.3")
        cb_test = ttk.Combobox(row3, textvariable=var_test, values=["0.2", "0.3", "0.4"], state="readonly", width=10)
        cb_test.pack(side=tk.LEFT, padx=(24, 8))

        # 4. PARAMETR: Walidacja krzyżowa
        cv_var = tk.BooleanVar(value=True)
        cv_row = tk.Frame(cf, bg=W98['bg'])
        cv_row.pack(fill=tk.X, pady=(8, 0))
        tk.Checkbutton(cv_row, text="Walidacja krzyżowa (5-fold CV) najlepszego modelu",
                       variable=cv_var, bg=W98['bg'], fg=W98['text'], font=W98['font'], activebackground=W98['bg']).pack(anchor=tk.W)

        def update_hint(*_):
            try: lbl_hint.config(text=f"Na gałąź: ~{max(1, int(var_exp.get()) // 3)}")
            except ValueError: lbl_hint.config(text="Na gałąź: —")
        var_exp.trace_add("write", update_hint)

        w98_separator(cf)

        def confirm():
            raw = var_exp.get().strip()
            if not raw or not raw.isdigit() or int(raw) < 29:
                messagebox.showwarning("Błąd", "Podaj liczbę całkowitą min 30.", parent=popup)
                return
            popup.destroy()
            self.run_experiments(int(raw), criterion=var_crit.get(), test_size=float(var_test.get()), do_cv=cv_var.get())

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
    def run_experiments(self, total=30, criterion='gini', test_size=0.3, do_cv=True):
        """
        Główna funkcja uruchamiająca wszystkie eksperymenty ML.

        Kroki:
            1. Oblicza liczbę eksperymentów na gałąź (total // 3) i zapamiętuje parametry.
            2. Generuje listę głębokości drzewa (depths) — równomiernie rozłożone od 1 do 30.
            3. Dla każdej z 3 gałęzi i każdej głębokości:
               - Trenuje DecisionTreeClassifier
               - Liczy metryki: Accuracy, Precision, Recall, F1 (train i test)
               - Wykrywa overfit (gdy train_acc - test_acc > 5%)
            4. Śledzi najlepszy model (najwyższa Accuracy na teście).
            5. Opcjonalnie uruchamia 5-fold CV na najlepszym modelu
               (z próbkowaniem do 100k rekordów dla wydajności).
            6. Wywołuje kolejno: _fill_results_table, _draw_comparison_charts,
               _draw_learning_curve, _update_best_model, _generate_auto_observations.

        Args:
            total (int): Całkowita liczba eksperymentów (dzielona na 3 gałęzie).
            criterion (str): Kryterium podziału drzewa: 'gini' lub 'entropy'.
            test_size (float): Ułamek danych przeznaczonych na zbiór testowy (0.2–0.4).
            do_cv (bool): Czy wykonać 5-fold cross-validation na najlepszym modelu.
        """
        n_per_branch = max(1, total // 3)
        actual_total = n_per_branch * 3

        self.last_total_exp = actual_total
        self.last_criterion = criterion
        self.last_test_size = test_size

        self.log(f"Start: {actual_total} eksperymentów. Kryterium: {criterion}, Test size: {test_size}...")
        self.set_status(f"Trwa {actual_total} eksperymentów...")
        self.all_results = []
        self.results_tree.delete(*self.results_tree.get_children())

        y = self.df.iloc[:, -1]
        depths = [5] if n_per_branch == 1 else sorted(set(np.linspace(1, 30, n_per_branch, dtype=int).tolist()))[:n_per_branch]

        exp_nr = 1
        best_acc = -1
        best_result, best_cm, best_model, best_features = None, None, None, None

        for branch_name, features in self.branches.items():
            X = self.df[features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            for depth in depths:
                pct = (exp_nr - 1) / actual_total * 100
                self.set_progress(pct, f"Exp {exp_nr}/{actual_total} — {branch_name}, depth={int(depth)}")

                model = DecisionTreeClassifier(max_depth=int(depth), criterion=criterion, random_state=42)
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
                    'train_acc': train_acc, 'overfit': overfit, 'classes': model.classes_
                }
                self.all_results.append(result)
                self.log(f"Exp {exp_nr:>3}/{actual_total} | {branch_name} depth={int(depth):>2} | Train={train_acc * 100:.1f}% Test={acc * 100:.1f}% {'⚠OVERFIT' if overfit else ''} Prec={prec * 100:.1f}% Rec={rec * 100:.1f}% F1={f1 * 100:.1f}%")

                if acc > best_acc:
                    best_acc, best_result, best_cm, best_model, best_features = acc, result, confusion_matrix(y_test, y_pred_test), model, features
                exp_nr += 1

        self.set_progress(100, "Zakończono!")
        self.log(f"=== ZAKOŃCZONO {actual_total} EKSPERYMENTÓW ===")
        
        cv_score = None
        if do_cv:
            self.log("Walidacja krzyżowa (5-fold) dla najlepszego modelu...")
            self.set_progress(100, "Cross-validation...")
            try:
                cv_model = DecisionTreeClassifier(max_depth=best_result['depth'], criterion=criterion, random_state=42)
                X_best, y_best = self.df[best_features], self.df.iloc[:, -1]
                if len(X_best) > 100_000:
                    idx = np.random.choice(len(X_best), 100_000, replace=False)
                    X_cv, y_cv = X_best.iloc[idx], y_best.iloc[idx]
                else:
                    X_cv, y_cv = X_best, y_best
                cv_scores = cross_val_score(cv_model, X_cv, y_cv, cv=5, scoring='accuracy')
                cv_score = cv_scores.mean()
                self.log(f"CV (5-fold): {cv_score * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
            except Exception as e:
                self.log(f"CV pominięta: {e}")

        self._fill_results_table(best_result)
        self._draw_comparison_charts()
        self._draw_learning_curve(best_result, best_features, test_size)
        self._update_best_model(best_result, best_cm, best_model, best_features, cv_score)
        self._generate_auto_observations()
        self.notebook.select(2)

    def _fill_results_table(self, best_result):
        """
        Wypełnia tabelę Treeview wynikami wszystkich eksperymentów z kolorowaniem.

        Kroki:
            1. Czyści poprzednią zawartość tabeli.
            2. Dla każdego wyniku przypisuje tag kolorystyczny:
               - 'best'    → najlepszy eksperyment (zielony)
               - 'overfit' → wykryty overfit (czerwonawy)
               - 'branch1/2/3' → kolor per gałąź
            3. Wstawia wiersz z wartościami (nr, gałąź, depth, metryki, overfit).

        Args:
            best_result (dict): Słownik wyników najlepszego eksperymentu.
        """
        self.results_tree.delete(*self.results_tree.get_children())
        tag_map = {'Gałąź 1': 'branch1', 'Gałąź 2': 'branch2', 'Gałąź 3': 'branch3'}
        for r in self.all_results:
            tag = 'best' if r['nr'] == best_result['nr'] else 'overfit' if r['overfit'] else tag_map.get(r['branch'], 'branch1')
            self.results_tree.insert('', tk.END, tags=(tag,), values=(
                r['nr'], r['branch'], r['depth'], f"{r['acc'] * 100:.2f}%", f"{r['prec'] * 100:.2f}%",
                f"{r['rec'] * 100:.2f}%", f"{r['f1'] * 100:.2f}%", f"{r['train_acc'] * 100:.2f}%", "⚠ TAK" if r['overfit'] else "OK"
            ))

    def _draw_comparison_charts(self):
        """
        Rysuje siatkę 6 wykresów porównawczych metryk dla wszystkich eksperymentów.

        Kroki:
            1. Tworzy siatkę 2×3 (GridSpec) w Matplotlib.
            2. Wykresy 1–4 (liniowe): Accuracy / Precision / Recall / F1 vs max_depth,
               osobna linia per gałąź.
            3. Wykres 5: Train vs Test Accuracy — linie ciągłe (test) i przerywane (train).
            4. Wykres 6 (słupkowy): średnie Train vs Test per gałąź — porównanie ogólne.
            5. Osadza całą figurę w self.compare_chart_frame.
        """
        for w in self.compare_chart_frame.winfo_children(): w.destroy()
        fig = plt.figure(figsize=(13, 7))
        fig.patch.set_facecolor('#c0c0c0')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

        metrics = [('acc', 'Accuracy (test)'), ('prec', 'Precision'), ('rec', 'Recall'), ('f1', 'F1-Score')]
        branch_colors = {'Gałąź 1': '#000080', 'Gałąź 2': '#008000', 'Gałąź 3': '#800000'}

        for idx, (mk, mn) in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.set_facecolor('#ffffff')
            for bn in self.branches:
                br = [r for r in self.all_results if r['branch'] == bn]
                ax.plot([r['depth'] for r in br], [r[mk] * 100 for r in br], marker='s', markersize=4, label=bn, color=branch_colors.get(bn, 'gray'), linewidth=1.5)
            ax.set_title(mn, fontsize=9, fontweight='bold')
            ax.set_xlabel("max_depth", fontsize=8)
            ax.set_ylabel("[%]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.4, linestyle='--')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor('#ffffff')
        for bn, color in branch_colors.items():
            br = sorted([r for r in self.all_results if r['branch'] == bn], key=lambda r: r['depth'])
            ax5.plot([r['depth'] for r in br], [r['train_acc'] * 100 for r in br], linestyle='--', color=color, linewidth=1.2, alpha=0.7, label=f"{bn} (train)")
            ax5.plot([r['depth'] for r in br], [r['acc'] * 100 for r in br], linestyle='-', color=color, linewidth=1.5, label=f"{bn} (test)")
        ax5.set_title("Train vs Test Accuracy", fontsize=9, fontweight='bold')
        ax5.legend(fontsize=6, ncol=2)
        ax5.grid(True, alpha=0.4, linestyle='--')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor('#ffffff')
        branch_names = list(self.branches.keys())
        avg_test = [np.mean([r['acc'] * 100 for r in self.all_results if r['branch'] == b]) for b in branch_names]
        avg_train = [np.mean([r['train_acc'] * 100 for r in self.all_results if r['branch'] == b]) for b in branch_names]
        x = np.arange(len(branch_names))
        ax6.bar(x - 0.2, avg_train, 0.35, label='Train', color=['#6699ff', '#66cc66', '#ff9999'], edgecolor='black', linewidth=0.5)
        ax6.bar(x + 0.2, avg_test, 0.35, label='Test', color=['#000080', '#008000', '#800000'], edgecolor='black', linewidth=0.5)
        ax6.set_xticks(x)
        ax6.set_xticklabels(['G1', 'G2', 'G3'], fontsize=8)
        ax6.set_title("Śr. Train vs Test per gałąź", fontsize=9, fontweight='bold')
        ax6.legend(fontsize=7)

        fig.suptitle("Wpływ max_depth i zestawu cech na mierniki jakości", fontsize=10, fontweight='bold')
        canvas = FigureCanvasTkAgg(fig, master=self.compare_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _draw_learning_curve(self, best_result, best_features, test_size):
        """
        Rysuje krzywą uczenia i wykres train/test vs głębokość dla najlepszego modelu.

        Kroki:
            1. Próbkuje dane do max 50k rekordów (optymalizacja wydajności).
            2. Generuje 8 rosnących rozmiarów zbioru treningowego (10% → 100%).
            3. Dla każdego rozmiaru trenuje model i mierzy accuracy (train i test).
            4. Lewy wykres: krzywa uczenia (accuracy vs liczba próbek treningowych)
               z zacieniowaną "luką overfittingu" między train a test.
            5. Prawy wykres: train vs test accuracy wg max_depth dla gałęzi najlepszego modelu,
               z zieloną pionową linią optymalnej głębokości.

        Args:
            best_result (dict): Słownik wyników najlepszego eksperymentu.
            best_features (list): Lista nazw cech najlepszego modelu.
            test_size (float): Ułamek danych testowych (użyty w train_test_split).
        """
        for w in self.learning_curve_frame.winfo_children(): w.destroy()
        X, y = self.df[best_features], self.df.iloc[:, -1]
        if len(X) > 50_000:
            idx = np.random.choice(len(X), 50_000, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]

        train_sizes_pct = np.linspace(0.1, 1.0, 8)
        train_sizes_abs = np.clip(np.floor(train_sizes_pct * len(X) * (1 - test_size)).astype(int), 10, None)
        train_accs, test_accs = [], []

        for ts in train_sizes_abs:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
            X_tr, y_tr = X_tr.iloc[:ts], y_tr.iloc[:ts]
            m = DecisionTreeClassifier(max_depth=best_result['depth'], criterion=self.last_criterion, random_state=42)
            m.fit(X_tr, y_tr)
            train_accs.append(accuracy_score(y_tr, m.predict(X_tr)) * 100)
            test_accs.append(accuracy_score(y_te, m.predict(X_te)) * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor('#c0c0c0')

        ax1.set_facecolor('#ffffff')
        ax1.plot(train_sizes_abs, train_accs, 'o-', color='#000080', label='Train', linewidth=2)
        ax1.plot(train_sizes_abs, test_accs, 's--', color='#800000', label='Test', linewidth=2)
        ax1.fill_between(train_sizes_abs, train_accs, test_accs, alpha=0.15, color='red', label='Luka overfittingu')
        ax1.set_title(f"Krzywa Uczenia — {best_result['branch']}, depth={best_result['depth']}", fontsize=10, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.4, linestyle='--')

        ax2.set_facecolor('#ffffff')
        best_branch_results = sorted([r for r in self.all_results if r['branch'] == best_result['branch']], key=lambda r: r['depth'])
        depths_list = [r['depth'] for r in best_branch_results]
        ax2.plot(depths_list, [r['train_acc'] * 100 for r in best_branch_results], 'o-', color='#000080', label='Train', linewidth=2)
        ax2.plot(depths_list, [r['acc'] * 100 for r in best_branch_results], 's--', color='#800000', label='Test', linewidth=2)
        ax2.fill_between(depths_list, [r['train_acc'] * 100 for r in best_branch_results], [r['acc'] * 100 for r in best_branch_results], alpha=0.15, color='red')
        ax2.axvline(x=best_result['depth'], color='green', linestyle=':', linewidth=1.5, label=f"Optymalny depth={best_result['depth']}")
        ax2.set_title(f"Train vs Test wg głębokości — {best_result['branch']}", fontsize=10, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.4, linestyle='--')

        fig.suptitle("Analiza overfittingu — krzywe uczenia", fontsize=11, fontweight='bold')
        canvas = FigureCanvasTkAgg(fig, master=self.learning_curve_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def _update_best_model(self, best, cm, model, features, cv_score=None):
        """
        Aktualizuje zakładkę 'Najlepszy Model' — etykiety metryk i wykresy.

        Kroki:
            1. Aktualizuje etykiety tekstowe: Accuracy, Precision, Recall, F1, CV score.
            2. Rysuje macierz pomyłek (ConfusionMatrixDisplay) z paletą Blues.
            3. Rysuje uproszczone drzewo decyzyjne (głębokość ograniczona do max 3
               poziomów dla czytelności wizualizacji).
            4. Osadza obie figury przez FigureCanvasTkAgg w odpowiednich ramkach.

        Args:
            best (dict): Słownik wyników najlepszego eksperymentu.
            cm (np.ndarray): Macierz pomyłek (confusion matrix).
            model (DecisionTreeClassifier): Wytrenowany najlepszy model.
            features (list): Lista nazw cech użytych w najlepszym modelu.
            cv_score (float|None): Wynik cross-validation lub None jeśli pominięto.
        """
        self.lbl_best_info.config(text=f"  Najlepszy: {best['branch']}, max_depth={best['depth']}  ")
        self.lbl_acc.config(text=f"  Accuracy:  {best['acc'] * 100:.2f}%  ")
        self.lbl_prec.config(text=f"  Precision: {best['prec'] * 100:.2f}%  ")
        self.lbl_rec.config(text=f"  Recall:    {best['rec'] * 100:.2f}%  ")
        self.lbl_f1.config(text=f"  F1-Score:  {best['f1'] * 100:.2f}%  ")
        if cv_score is not None: self.lbl_cv.config(text=f"  CV (5-fold): {cv_score * 100:.2f}%  ")

        for w in self.cm_frame.winfo_children(): w.destroy()
        fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
        fig1.patch.set_facecolor('#c0c0c0')
        ax1.set_facecolor('#ffffff')
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best['classes']).plot(cmap='Blues', ax=ax1, values_format='d')
        ax1.set_title("Macierz Pomyłek", fontsize=9)
        fig1.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig1, master=self.cm_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig1)

        for w in self.tree_frame.winfo_children(): w.destroy()
        try:
            depth_vis = min(best['depth'], 3)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#c0c0c0')
            ax2.set_facecolor('#ffffff')
            plot_tree(model, max_depth=depth_vis, feature_names=features, class_names=[str(c) for c in best['classes']], filled=True, rounded=True, fontsize=6, ax=ax2)
            ax2.set_title(f"Drzewo decyzyjne (depth ≤ {depth_vis})", fontsize=9)
            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=self.tree_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig2)
        except Exception as e:
            self.log(f"Wizualizacja drzewa niedostępna: {e}")

    def _generate_auto_observations(self):
        """
        Generuje automatyczny raport tekstowy z podsumowaniem wszystkich eksperymentów.

        Kroki:
            1. Wyszukuje najlepszy i najgorszy wynik po Accuracy.
            2. Wypisuje metryki najlepszego (Acc, Prec, Rec, F1, Train, Overfit).
            3. Wypisuje metryki najgorszego (Acc, F1).
            4. Liczy łączną liczbę eksperymentów z overfittingiem.
            5. Dla każdej gałęzi oblicza: średnie Test Acc, Train Acc i lukę (gap).
            6. Wstawia gotowy tekst do self.auto_obs_text i przełącza na zakładkę Obserwacje.
        """
        if not self.all_results: return
        lines = ["=" * 60, "  AUTOMATYCZNE SPOSTRZEZENIA Z EKSPERYMENTOW", "=" * 60, ""]
        best = max(self.all_results, key=lambda r: r['acc'])
        worst = min(self.all_results, key=lambda r: r['acc'])

        lines.extend([
            f"[+] Najlepszy: {best['branch']}, depth={best['depth']}",
            f"    Acc={best['acc'] * 100:.2f}%  Prec={best['prec'] * 100:.2f}%  Rec={best['rec'] * 100:.2f}%  F1={best['f1'] * 100:.2f}%",
            f"    Train={best['train_acc'] * 100:.2f}%  Overfit: {'TAK' if best['overfit'] else 'NIE'}\n",
            f"[-] Najgorszy: {worst['branch']}, depth={worst['depth']}",
            f"    Acc={worst['acc'] * 100:.2f}%  F1={worst['f1'] * 100:.2f}%\n",
            f"[!] Eksperymenty z overfittingiem (>5%): {sum(1 for r in self.all_results if r['overfit'])}/{len(self.all_results)}\n",
            "-" * 60, "  SREDNIE PER GALAZ:", "-" * 60
        ])

        for branch in self.branches:
            br = [r for r in self.all_results if r['branch'] == branch]
            avg_acc, avg_train = np.mean([r['acc'] for r in br]) * 100, np.mean([r['train_acc'] for r in br]) * 100
            lines.extend([
                f"  {branch} ({len(self.branches[branch])} cech):",
                f"    Test Acc={avg_acc:.2f}%  Train={avg_train:.2f}%  Gap={avg_train - avg_acc:.2f}%"
            ])

        lines.extend(["", "=" * 60, "  Uzupelnij wnioski w polach po prawej stronie.", "=" * 60])
        self.auto_obs_text.config(state=tk.NORMAL)
        self.auto_obs_text.delete('1.0', tk.END)
        self.auto_obs_text.insert(tk.END, "\n".join(lines))
        self.auto_obs_text.config(state=tk.DISABLED)
        self.notebook.select(6)

    # ─────────────────────────────────────────────────────────────────────
    # EKSPORT
    # ─────────────────────────────────────────────────────────────────────
    def export_results_csv(self):
        """
        Eksportuje surowe wyniki wszystkich eksperymentów do pliku CSV.

        Kroki:
            1. Sprawdza czy są wyniki — jeśli nie, pokazuje ostrzeżenie.
            2. Otwiera dialog zapisu pliku (domyślna nazwa: wyniki.csv).
            3. Zapisuje każdy wynik jako wiersz CSV z kolumnami:
               nr, branch, depth, acc, prec, rec, f1, train_acc, overfit (TAK/NIE).
               Wartości metryczne formatowane do 4 miejsc po przecinku.
        """
        if not self.all_results: return messagebox.showwarning("Brak danych", "Najpierw uruchom eksperymenty!")
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], initialfile="wyniki.csv")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['nr', 'branch', 'depth', 'acc', 'prec', 'rec', 'f1', 'train_acc', 'overfit'])
                writer.writeheader()
                for r in self.all_results:
                    writer.writerow({
                        'nr': r['nr'], 'branch': r['branch'], 'depth': r['depth'],
                        'acc': f"{r['acc']:.4f}", 'prec': f"{r['prec']:.4f}", 'rec': f"{r['rec']:.4f}",
                        'f1': f"{r['f1']:.4f}", 'train_acc': f"{r['train_acc']:.4f}", 'overfit': 'TAK' if r['overfit'] else 'NIE'
                    })
            self.log(f"Wyniki zapisane: {path}")
        except Exception as e: messagebox.showerror("Błąd", str(e))

    def export_observations_txt(self):
        """
        Eksportuje pełny raport tekstowy z wynikami i wnioskami użytkownika.

        Kroki:
            1. Sprawdza czy są wyniki — jeśli nie, pokazuje ostrzeżenie.
            2. Otwiera dialog zapisu pliku (domyślna nazwa: raport.txt).
            3. Pobiera treść pól tekstowych: obs_text, wn_text, end_text.
            4. Zapisuje raport w 4 sekcjach:
               - Dane (rekordy, cechy, kolumna docelowa)
               - Parametry startowe (criterion, test_size, liczba eksperymentów)
               - Wynik najlepszego modelu (branch, depth, metryki, overfit)
               - Własne spostrzeżenia i wnioski użytkownika (z pól tekstowych)
        """
        if not self.all_results:
            return messagebox.showwarning("Uwaga", "Brak wyników do zapisu!")
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")], initialfile="raport.txt")
        if not path:
            return

        best = max(self.all_results, key=lambda r: r['acc'])
        obs, wn, end = self.obs_text.get('1.0', tk.END).strip(), self.wn_text.get('1.0', tk.END).strip(), self.end_text.get('1.0', tk.END).strip()

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n  RAPORT Z EKSPERYMENTÓW\n" + "=" * 60 + "\n\n")
                f.write("1. DANE\n" + "-" * 30 + "\n")
                f.write(f"Rekordy: {self._meta['rows']}\nCechy startowe: {self._meta['features']}\nKolumna docelowa (cel): {self._meta['target']}\n\n")
                f.write("2. PARAMETRY STARTOWE\n" + "-" * 30 + "\n")
                f.write(f"Wykonanych eksperymentów: {self.last_total_exp}\nKryterium podziału (criterion): {self.last_criterion}\nRozmiar zbioru testowego (test_size): {self.last_test_size}\n\n")
                f.write("3. WYNIK I OCENA NAJLEPSZEGO MODELU\n" + "-" * 30 + "\n")
                f.write(f"Struktura wejściowa: {best['branch']}\nOptymalny koszt (max_depth): {best['depth']}\n")
                f.write(f"Accuracy:  {best['acc']*100:.2f}%\nPrecision: {best['prec']*100:.2f}%\nRecall:    {best['rec']*100:.2f}%\nF1-Score:  {best['f1']*100:.2f}%\nOverfit:   {'TAK' if best['overfit'] else 'NIE'}\n\n")
                f.write("4. WŁASNE SPOSTRZEŻENIA I WNIOSKI UŻYTKOWNIKA\n" + "-" * 30 + "\n")
                f.write(f"[Obserwacje]:\n{obs if obs else '---'}\n\n[Wnioski na podstawie obserwacji]:\n{wn if wn else '---'}\n\n[Wnioski końcowe]:\n{end if end else '---'}\n")
            self.log(f"Zapisano szczegółowy raport do: {path}")
        except Exception as e: messagebox.showerror("Błąd", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MLProjectGUI(root)
    root.mainloop()
