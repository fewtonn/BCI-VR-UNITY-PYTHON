from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
import sys
import numpy as np
import pandas as pd # <--- NOVA BIBLIOTECA NECESSÁRIA
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pylsl import StreamInlet, resolve_byprop
import threading
import zmq
import socket
import time
import random

# --- TENTA IMPORTAR O CONFIG ---
try:
    from config import GABARITO_SESSAO, PORCENTAGEM_TL, EPOCHS_TREINO
except ImportError:
    GABARITO_SESSAO = [0, 1, 2] * 50
    PORCENTAGEM_TL = 0.2
    EPOCHS_TREINO = 1

# --- CONFIGURAÇÕES GLOBAIS ---
USAR_MODELO = True 
PORTA_UNITY = 5555
PORTA_UDP_UNITY = 12346

if USAR_MODELO:
    try:
        from keras.models import load_model, Sequential
        from keras.layers import Input, Dense
        from keras.optimizers import Adam
        import os
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    except ImportError:
        USAR_MODELO = False

# =============================================================================
# CLASSE DE CONEXÃO COM UNITY
# =============================================================================
class UnitySender:
    def __init__(self, port=PORTA_UNITY, udp_port=PORTA_UDP_UNITY):
        self.port = port
        self.udp_port = udp_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        try: self.socket.setsockopt(zmq.CONFLATE, 1)
        except: pass 
        
        try: self.socket.bind(f"tcp://*:{port}")
        except zmq.ZMQError: pass 

        self.local_ip = self.get_local_ip()
        self.send_ip_udp_broadcast()
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.thread.start()

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except: return "127.0.0.1"

    def send_ip_udp_broadcast(self):
        def _broadcast():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self.running:
                try:
                    s.sendto(self.local_ip.encode(), ('<broadcast>', self.udp_port))
                    time.sleep(1.0) 
                except: pass
            s.close()
        threading.Thread(target=_broadcast, daemon=True).start()

    def send(self, msg):
        with self.lock: self.queue.append(msg)

    def sender_loop(self):
        while self.running:
            with self.lock:
                if self.queue:
                    try: self.socket.send_string(str(self.queue.pop(0)))
                    except: pass
            time.sleep(0.001) 

    def stop(self):
        self.running = False
        try: self.socket.close(); self.context.term()
        except: pass

# =============================================================================
# JANELA PRINCIPAL
# =============================================================================
class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1250, 850) 
        self.setWindowTitle('BCI Control Center')
        self.aplicar_estilo_escuro()

        # --- Variáveis de Sistema ---
        self.unity = None
        self.inlet = None
        self.model = None
        self.conectado_unity = False
        self.sessao_iniciada = False
        self.sincronizado = False
        self.modo_teste_unity = False 
        
        # --- Variáveis de Arquivo (NOVO) ---
        self.modo_arquivo = False
        self.dados_arquivo = None
        self.ponteiro_arquivo = 0
        
        # --- Configs Hardware ---
        self.canais = ['C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8','T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
        self.n_channels_hardware = len(self.canais) 
        self.x_size = 500 
        
        # --- BUFFER DA ESTEIRA ---
        self.buffer_sobra = [] 
        
        # --- CONTROLE DA SESSÃO ---
        self.indice_atual = 0
        self.total_tentativas = len(GABARITO_SESSAO)
        self.qtd_tl = int(self.total_tentativas * PORCENTAGEM_TL)
        self.acertos_fase1 = 0
        self.acertos_fase2 = 0
        
        # --- VISUALIZAÇÃO ---
        self.current_data_visual = np.zeros((self.x_size, self.n_channels_hardware))
        self.fs = 250.0  
        self.escala_visual = 150 
        self.escala_auto = False
        self.fft_smooth_factor = 0.0
        self.fft_buffer_history = np.zeros((self.n_channels_hardware, self.x_size//2))

        # --- LAYOUT ---
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        
        self.panel_left = QFrame()
        self.panel_left.setFixedWidth(320)
        self.layout_left = QVBoxLayout(self.panel_left)
        self.setup_painel_esquerdo()
        self.main_layout.addWidget(self.panel_left)

        self.panel_right = QWidget()
        self.layout_right = QVBoxLayout(self.panel_right)
        self.tabs = QTabWidget()
        self.setup_tabs()
        self.layout_right.addWidget(self.tabs)
        self.main_layout.addWidget(self.panel_right, 1)

        self.setup_menu()

    def setup_painel_esquerdo(self):
        lbl_titulo = QLabel("CONTROLES")
        lbl_titulo.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_left.addWidget(lbl_titulo)

        # Status
        group_conn = QGroupBox("Conexões e Arquivos")
        form_conn = QFormLayout()
        self.lbl_lsl = QLabel("Desconectado"); self.lbl_lsl.setStyleSheet("color: #ff5555;")
        self.lbl_csv = QLabel("Nenhum"); self.lbl_csv.setStyleSheet("color: gray;") # NOVO LABEL
        self.lbl_unity = QLabel("Desconectado"); self.lbl_unity.setStyleSheet("color: #ff5555;")
        self.lbl_model = QLabel("Nenhum"); self.lbl_model.setStyleSheet("color: gray;")
        form_conn.addRow("LSL:", self.lbl_lsl)
        form_conn.addRow("CSV:", self.lbl_csv) # ADICIONADO NO PAINEL
        form_conn.addRow("Unity:", self.lbl_unity)
        form_conn.addRow("IA:", self.lbl_model)
        group_conn.setLayout(form_conn)
        self.layout_left.addWidget(group_conn)

        # --- SHAPE MANUAL ---
        group_shape = QGroupBox("Shape do Modelo")
        layout_shape = QFormLayout()
        self.spin_shape_time = QSpinBox(); self.spin_shape_time.setRange(10, 5000); self.spin_shape_time.setValue(721); self.spin_shape_time.setSuffix(" pts")
        self.spin_shape_ch = QSpinBox(); self.spin_shape_ch.setRange(1, 32); self.spin_shape_ch.setValue(16); self.spin_shape_ch.setSuffix(" ch")
        layout_shape.addRow("Time Steps:", self.spin_shape_time)
        layout_shape.addRow("Canais:", self.spin_shape_ch)
        group_shape.setLayout(layout_shape)
        self.layout_left.addWidget(group_shape)

        # --- CONTROLES E BOTÕES ---
        self.layout_left.addSpacing(10)
        
        self.chk_teste_unity = QCheckBox("Modo Teste Unity (Aleatório)")
        self.chk_teste_unity.setStyleSheet("color: #ff9800; font-weight: bold;")
        self.chk_teste_unity.setToolTip("Gera sinais aleatórios para testar o Unity.")
        self.layout_left.addWidget(self.chk_teste_unity)

        # --- NOVOS BOTÕES ---
        self.btn_csv = QPushButton("📁 1. Carregar CSV (Simulação)"); self.btn_csv.clicked.connect(self.carregar_csv)
        self.btn_lsl = QPushButton("📡 2. Conectar LSL (Tempo Real)"); self.btn_lsl.clicked.connect(self.conectar_LSL)
        self.btn_unity = QPushButton("🎮 3. Conectar Unity"); self.btn_unity.clicked.connect(self.conectarUnity)
        self.btn_iniciar = QPushButton("▶ 4. Iniciar Sessão"); self.btn_iniciar.setStyleSheet("background-color: #2e7d32; font-weight: bold;")
        self.btn_iniciar.clicked.connect(self.iniciar_sessao)
        
        self.layout_left.addWidget(self.btn_csv)
        self.layout_left.addWidget(self.btn_lsl)
        self.layout_left.addWidget(self.btn_unity)
        self.layout_left.addWidget(self.btn_iniciar)

        # Progresso
        group_prog = QGroupBox("Progresso")
        layout_prog = QVBoxLayout()
        self.lbl_progresso = QLabel("0 / 0"); self.lbl_progresso.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_fase = QLabel("AGUARDANDO"); self.lbl_fase.setStyleSheet("color: yellow; font-weight: bold;"); self.lbl_fase.setAlignment(QtCore.Qt.AlignCenter)
        self.bar_progresso = QProgressBar(); self.bar_progresso.setValue(0)
        layout_prog.addWidget(self.lbl_progresso); layout_prog.addWidget(self.bar_progresso); layout_prog.addWidget(self.lbl_fase)
        group_prog.setLayout(layout_prog)
        self.layout_left.addWidget(group_prog)

        # Resultado
        group_res = QGroupBox("Predição Atual")
        layout_res = QVBoxLayout()
        self.lbl_predicao = QLabel("--"); self.lbl_predicao.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold)); self.lbl_predicao.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_feedback = QLabel(""); self.lbl_feedback.setAlignment(QtCore.Qt.AlignCenter)
        layout_res.addWidget(self.lbl_predicao); layout_res.addWidget(self.lbl_feedback)
        group_res.setLayout(layout_res)
        self.layout_left.addWidget(group_res)

        self.layout_left.addStretch()

    def aplicar_estilo_escuro(self):
        qss = """
        QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', Arial; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; background-color: #2b2b2b; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #2b2b2b; color: #aaaaaa; }
        QPushButton { background-color: #3c3f41; border: 1px solid #555; border-radius: 4px; padding: 5px; color: white; }
        QPushButton:hover { background-color: #484b4d; }
        QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
        QTabBar::tab { background: #2b2b2b; color: #888888; padding: 8px 25px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: bold; }
        QTabBar::tab:selected { background: #3c3f41; color: #ffffff; border-bottom: 3px solid #00bcd4; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: #3c3f41; border: 1px solid #555; padding: 3px; color: white; }
        QProgressBar { border: 1px solid #555; text-align: center; color: white; }
        QProgressBar::chunk { background-color: #00bcd4; }
        QCheckBox { color: white; spacing: 5px; }
        QCheckBox::indicator { width: 15px; height: 15px; }
        """
        self.setStyleSheet(qss)

    def setup_tabs(self):
        self.tab_time = QWidget()
        l_time = QVBoxLayout(self.tab_time)
        tb_time = QHBoxLayout()
        self.combo_scale = QComboBox(); self.combo_scale.addItems(["Auto", "50 uV", "100 uV", "200 uV", "400 uV"])
        self.combo_scale.setCurrentText("200 uV")
        self.combo_scale.currentTextChanged.connect(lambda t: setattr(self, 'escala_auto', True) if t=="Auto" else (setattr(self, 'escala_auto', False), setattr(self, 'escala_visual', int(t.split()[0])), self.atualizar_limites_temporal()))
        tb_time.addWidget(QLabel("Escala:")); tb_time.addWidget(self.combo_scale); tb_time.addStretch()
        l_time.addLayout(tb_time)

        self.fig_time = Figure(figsize=(5,3), dpi=100, facecolor='#ffffff')
        self.can_time = FigureCanvas(self.fig_time)
        self.setup_grafico_temporal()
        l_time.addWidget(self.can_time)
        self.tabs.addTab(self.tab_time, "Série Temporal")

        self.tab_fft = QWidget()
        l_fft = QVBoxLayout(self.tab_fft)
        tb_fft = QHBoxLayout()
        self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 0.99); self.spin_smooth.setSingleStep(0.1)
        self.spin_smooth.valueChanged.connect(lambda: setattr(self, 'fft_smooth_factor', self.spin_smooth.value()))
        tb_fft.addWidget(QLabel("Smooth:")); tb_fft.addWidget(self.spin_smooth); tb_fft.addStretch()
        l_fft.addLayout(tb_fft)

        self.fig_fft = Figure(figsize=(5,3), dpi=100, facecolor='#ffffff')
        self.can_fft = FigureCanvas(self.fig_fft)
        self.setup_grafico_fft()
        l_fft.addWidget(self.can_fft)
        self.tabs.addTab(self.tab_fft, "FFT")

    def setup_grafico_temporal(self):
        self.ax_time = self.fig_time.add_subplot(111)
        self.fig_time.patch.set_facecolor('#ffffff'); self.ax_time.set_facecolor('#ffffff')
        self.ax_time.tick_params(colors='#333333'); self.ax_time.set_xlim(0, self.x_size); self.ax_time.set_yticks([])
        for spine in self.ax_time.spines.values(): spine.set_color('#aaaaaa')
        colors = ['#555555', '#8959a8', '#3e999f', '#71c671', '#e8c346', '#e68136', '#d84e4e', '#8c564b']
        self.lines_time = []; self.rms_texts = []
        for i in range(self.n_channels_hardware):
            l, = self.ax_time.plot([],[], lw=1.2, color=colors[i%8])
            self.lines_time.append(l)
            self.rms_texts.append(self.ax_time.text(self.x_size+10, 0, "", fontsize=9, color='#333333'))
        self.atualizar_limites_temporal()

    def setup_grafico_fft(self):
        self.ax_fft = self.fig_fft.add_subplot(111)
        self.fig_fft.patch.set_facecolor('#ffffff'); self.ax_fft.set_facecolor('#ffffff')
        self.ax_fft.tick_params(colors='#333333', which='both'); self.ax_fft.set_yscale('log')
        self.ax_fft.set_ylim(0.1, 100); self.ax_fft.set_xlim(0, 60)
        self.ax_fft.grid(True, which='both', color='#dddddd', alpha=0.8)
        self.ax_fft.set_xlabel('Freq (Hz)', color='#555555'); self.ax_fft.set_ylabel('uV', color='#555555')
        for spine in self.ax_fft.spines.values(): spine.set_color('#aaaaaa')
        colors = ['#555555', '#8959a8', '#3e999f', '#71c671', '#e8c346', '#e68136', '#d84e4e', '#8c564b']
        self.lines_fft = [self.ax_fft.plot([],[], lw=1.5, alpha=0.8, color=colors[i%8])[0] for i in range(self.n_channels_hardware)]

    def atualizar_limites_temporal(self):
        top = self.n_channels_hardware * self.escala_visual
        self.ax_time.set_ylim(-self.escala_visual, top + self.escala_visual)

    # =========================================================================
    # NOVA FUNÇÃO DE CARREGAMENTO DE ARQUIVO CSV
    # =========================================================================
    def carregar_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir CSV OpenBCI', '', "CSV (*.csv)")
        if fname:
            try:
                # O parâmetro comment='%' ignora automaticamente o cabeçalho do OpenBCI
                df = pd.read_csv(fname, comment='%')
                
                # Assume que a coluna 0 é o Index da amostra e as colunas de 1 a n_channels_hardware são os dados EXG
                self.dados_arquivo = df.iloc[:, 1 : self.n_channels_hardware + 1].values
                
                self.ponteiro_arquivo = 0
                self.modo_arquivo = True
                
                nome_arq = fname.split('/')[-1]
                self.lbl_csv.setText(f"{nome_arq} ({len(self.dados_arquivo)} pts)")
                self.lbl_csv.setStyleSheet("color: #00e676;")
                self.btn_csv.setEnabled(False)
                
                QMessageBox.information(self, "Sucesso", "Arquivo CSV OpenBCI carregado e pronto para streaming.")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao ler o arquivo CSV:\n{str(e)}")

    def conectarUnity(self):
        if self.conectado_unity: return
        try: self.unity = UnitySender(); self.conectado_unity = True; self.lbl_unity.setText("Conectado"); self.lbl_unity.setStyleSheet("color: #00e676;"); self.btn_unity.setEnabled(False)
        except Exception as e: QMessageBox.critical(self, "Erro", str(e))

    def conectar_LSL(self):
        self.lbl_lsl.setText("Procurando..."); QApplication.processEvents()
        streams = resolve_byprop('type', 'EEG', timeout=3)
        if streams:
            self.inlet = StreamInlet(streams[0])
            self.lbl_lsl.setText(f"Conectado ({streams[0].channel_count()})"); self.lbl_lsl.setStyleSheet("color: #00e676;"); self.btn_lsl.setEnabled(False)
        else: self.lbl_lsl.setText("Erro"); QMessageBox.warning(self, "Erro", "LSL não encontrado")

    def setup_menu(self):
        self.menuBar().addMenu("Arquivo").addAction("Carregar Modelo").triggered.connect(self.carregar_modelo_arquivo)

    def carregar_modelo_arquivo(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Modelo', '', "H5 (*.h5)")
        if fname and USAR_MODELO:
            try:
                old = load_model(fname)
                if old.output_shape[-1] != 3:
                    new = Sequential()
                    new.add(Input(shape=old.input_shape[1:]))
                    for l in old.layers[:-1]: new.add(l)
                    new.add(Dense(3, activation='softmax'))
                    self.model = new
                else: self.model = old
                self.model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.lbl_model.setText(fname.split('/')[-1]); self.lbl_model.setStyleSheet("color: #00e676;")
            except Exception as e: QMessageBox.critical(self, "Erro", str(e))

    def iniciar_sessao(self):
        self.modo_teste_unity = self.chk_teste_unity.isChecked()
        
        # --- VERIFICAÇÃO ALTERADA ---
        if self.modo_teste_unity:
            if not self.conectado_unity:
                ret = QMessageBox.question(self, "Unity não conectado", "Deseja iniciar o teste aleatório mesmo assim?", QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.No: return
        else:
            # Agora ele checa se tem o LSL OU se tem um arquivo carregado
            if not self.inlet and not self.modo_arquivo: 
                return QMessageBox.warning(self, "Aviso", "Conecte o LSL, carregue um CSV ou ative o Modo Teste Aleatório!")
        
        self.sessao_iniciada = True
        self.btn_iniciar.setEnabled(False)
        self.bar_progresso.setMaximum(self.total_tentativas)
        self.bar_progresso.setValue(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(10) # Roda a cada 10ms

    def update_loop(self):
        if self.modo_teste_unity:
            data = np.random.randn(3, self.n_channels_hardware) * 50 
            self.sincronizado = True
            self.buffer_sobra.extend(data)
            self.current_data_visual = np.roll(self.current_data_visual, -3, axis=0)
            self.current_data_visual[-3:, :] = data
            
        # ==============================================================
        # NOVA LÓGICA DE INJEÇÃO VIA ARQUIVO (Simula streaming LSL)
        # ==============================================================
        elif self.modo_arquivo:
            chunk_size = 3 # 3 amostras a cada 10ms = aprox 250Hz a 300Hz real
            if self.ponteiro_arquivo + chunk_size < len(self.dados_arquivo):
                data = self.dados_arquivo[self.ponteiro_arquivo : self.ponteiro_arquivo + chunk_size]
                self.ponteiro_arquivo += chunk_size
                
                self.sincronizado = True
                self.buffer_sobra.extend(data)
                self.current_data_visual = np.roll(self.current_data_visual, -len(data), axis=0)
                self.current_data_visual[-len(data):, :] = data
            else:
                self.finalizar_sessao()
                return QMessageBox.information(self, "Fim do Arquivo", "A leitura do CSV chegou ao final.")
        
        # Lógica Original LSL
        else:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0)
            if chunk:
                data = np.array(chunk)[:, :self.n_channels_hardware]
                if not self.sincronizado:
                    if np.sum(np.abs(data)) > 1e-6: self.sincronizado = True
                    else: return
                self.buffer_sobra.extend(data)
                self.current_data_visual = np.roll(self.current_data_visual, -len(data), axis=0)
                self.current_data_visual[-len(data):, :] = data

        self.atualizar_graficos_visuais()

        target_time = self.spin_shape_time.value()
        target_ch = self.spin_shape_ch.value()
        
        while len(self.buffer_sobra) >= target_time:
            if self.modo_teste_unity:
                if self.indice_atual >= self.total_tentativas: self.indice_atual = 0
            elif self.indice_atual >= self.total_tentativas: 
                self.finalizar_sessao()
                return
            
            raw_epoch = np.array(self.buffer_sobra[:target_time])
            self.buffer_sobra = self.buffer_sobra[target_time:] 
            processed_epoch = raw_epoch[:, :target_ch]
            
            self.processar_caixa(processed_epoch)

    def processar_caixa(self, dados):
        pred = 2
        label_real = GABARITO_SESSAO[self.indice_atual]

        if self.modo_teste_unity:
            pred = random.randint(0, 2)
            self.lbl_fase.setText("MODO TESTE (Sem IA)")
            self.lbl_fase.setStyleSheet("color: orange; font-weight: bold;")
        else:
            dados_norm = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8)
            input_data = np.expand_dims(dados_norm, axis=0).astype(np.float32)
            
            if self.model:
                try:
                    prob = self.model.predict(input_data, verbose=0)[0]
                    if len(prob) == 3: pred = np.argmax(prob)
                    else: pred = 0 if prob[0] < 0.4 else (1 if prob[0] > 0.6 else 2)
                except: pass
            
            fase = "TREINO (TL)" if self.indice_atual < self.qtd_tl else "TESTE"
            if self.modo_arquivo: fase += " [ARQUIVO]"
            self.lbl_fase.setText(fase)
            self.lbl_fase.setStyleSheet(f"color: {'yellow' if 'TREINO' in fase else '#00e676'}; font-weight: bold;")

        acertou = (pred == label_real)
        self.indice_atual += 1
        
        self.lbl_progresso.setText(f"{self.indice_atual}/{self.total_tentativas}")
        self.bar_progresso.setValue(self.indice_atual)
        
        nomes = ["ESQUERDA", "DIREITA", "REPOUSO"]
        cores = ["#00bcd4", "#ff4081", "white"]
        self.lbl_predicao.setText(nomes[pred])
        self.lbl_predicao.setStyleSheet(f"color: {cores[pred]}")
        
        self.lbl_feedback.setText("ACERTOU" if acertou else "ERROU")
        self.lbl_feedback.setStyleSheet(f"color: {'#00e676' if acertou else '#ff5555'}")

        if self.conectado_unity:
            if acertou:
                if pred == 0: self.unity.send("LEFT")
                elif pred == 1: self.unity.send("RIGHT")
                else: self.unity.send("REST")
            else:
                self.unity.send("REST")

        if not self.modo_teste_unity and self.indice_atual < self.qtd_tl and self.model:
            if acertou: self.acertos_fase1 += 1
            d_norm = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8)
            inp = np.expand_dims(d_norm, axis=0).astype(np.float32)
            self.model.train_on_batch(inp, np.array([label_real]).astype(np.float32))
        elif not self.modo_teste_unity:
            if acertou: self.acertos_fase2 += 1

    def finalizar_sessao(self):
        self.timer.stop(); self.btn_iniciar.setEnabled(True); self.sessao_iniciada = False
        if not self.modo_teste_unity:
            acc = (self.acertos_fase2 / (self.total_tentativas - self.qtd_tl))*100 if (self.total_tentativas - self.qtd_tl) > 0 else 0
            QMessageBox.information(self, "Fim", f"Acurácia Teste: {acc:.2f}%")

    def atualizar_graficos_visuais(self):
        if self.tabs.currentIndex() == 0: 
            if self.escala_auto:
                amp = np.ptp(self.current_data_visual, axis=0).max()
                if amp > 1: self.escala_visual = amp * 0.8; self.atualizar_limites_temporal()
            x = np.arange(self.x_size)
            for i, l in enumerate(self.lines_time):
                off = i * self.escala_visual
                y = self.current_data_visual[:, i] - np.mean(self.current_data_visual[:, i])
                l.set_data(x, y + off)
                rms = np.sqrt(np.mean(y**2))
                self.rms_texts[i].set_text(f"{rms:.2f} uVrms"); self.rms_texts[i].set_position((self.x_size+10, off))
            self.can_time.draw_idle()
        elif self.tabs.currentIndex() == 1: 
            xf = np.linspace(0, self.fs/2, self.x_size//2)
            for i, l in enumerate(self.lines_fft):
                raw = 2.0/self.x_size * np.abs(fft(self.current_data_visual[:, i])[0:self.x_size//2])
                f = self.fft_smooth_factor
                self.fft_buffer_history[i] = (self.fft_buffer_history[i]*f) + (raw*(1-f))
                l.set_data(xf, self.fft_buffer_history[i])
            self.can_fft.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = JanelaInicial()
    win.show()
    sys.exit(app.exec_())