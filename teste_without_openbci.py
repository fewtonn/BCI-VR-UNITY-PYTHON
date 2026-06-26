from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygon
import sys
import numpy as np
import pandas as pd
from scipy.fft import fft
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pylsl import StreamInlet, resolve_byprop
import threading
import zmq
import socket
import time
import random
import os
from datetime import datetime

# =============================================================================
# BLINDAGEM DE VARIÁVEIS GLOBAIS
# =============================================================================
EPOCHS_TREINO = 1
USAR_MODELO = True 
PORTA_UNITY = 5555
PORTA_UDP_UNITY = 12346

try:
    import config
    if hasattr(config, 'EPOCHS_TREINO'): EPOCHS_TREINO = config.EPOCHS_TREINO
except Exception: pass

if USAR_MODELO:
    try:
        from keras.models import load_model, Sequential
        from keras.layers import Input, Dense
        from keras.optimizers import Adam
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    except ImportError:
        USAR_MODELO = False
        print("Aviso: Keras/TensorFlow não instalados.")

# =============================================================================
# GAUGE WIDGET (Velocímetro de Confiança)
# =============================================================================
class GaugeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(250, 130)
        self.current_angle = 0.0; self.base_angle = 0.0; self.target_angle = 0.0; self.incerteza = 0.0    
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self.update_animation); self.anim_timer.start(30) 

    def set_probabilities(self, prob_left, prob_right, prob_rest):
        self.target_angle = (prob_right * 60) + (prob_left * -60)
        self.incerteza = 1.0 - max(prob_left, prob_right, prob_rest)

    def update_animation(self):
        diff = self.target_angle - self.base_angle
        if abs(diff) > 0.1: self.base_angle += diff * 0.15
        self.current_angle = self.base_angle + (random.uniform(-1.0, 1.0) * (self.incerteza * 15.0))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); cx, cy = w / 2, h - 20; r = min(w / 2, h) - 25
        rect = QRectF(cx - r, cy - r, r * 2, r * 2)
        
        painter.setPen(QPen(QColor("#00bcd4"), 15, Qt.SolidLine, Qt.FlatCap)); painter.drawArc(rect, 120 * 16, 60 * 16) 
        painter.setPen(QPen(QColor("#888888"), 15, Qt.SolidLine, Qt.FlatCap)); painter.drawArc(rect, 60 * 16, 60 * 16)
        painter.setPen(QPen(QColor("#ff4081"), 15, Qt.SolidLine, Qt.FlatCap)); painter.drawArc(rect, 0 * 16, 60 * 16)

        painter.setPen(QColor("#ffffff")); painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        painter.drawText(int(cx - r - 20), int(cy), "ESQ"); painter.drawText(int(cx + r + 0), int(cy), "DIR")
        painter.translate(cx, cy); painter.rotate(self.current_angle)
        
        painter.setPen(Qt.NoPen); painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawPolygon(QPolygon([QPoint(-4, 0), QPoint(4, 0), QPoint(0, int(-r + 5))]))
        painter.setBrush(QBrush(QColor("#ffc107"))); painter.drawEllipse(QPoint(0, 0), 6, 6)

# =============================================================================
# COMUNICAÇÃO COM O UNITY (ZMQ)
# =============================================================================
class UnitySender:
    def __init__(self, port=PORTA_UNITY, udp_port=PORTA_UDP_UNITY):
        self.port = port; self.udp_port = udp_port; self.context = zmq.Context(); self.socket = self.context.socket(zmq.PUB)
        try: self.socket.setsockopt(zmq.CONFLATE, 1)
        except Exception: pass 
        try: self.socket.bind(f"tcp://*:{port}")
        except Exception as e: print(f"Erro ao ligar a porta ZMQ: {e}") 

        self.local_ip = self.get_local_ip()
        self.send_ip_udp_broadcast()
        self.queue = []; self.lock = threading.Lock(); self.running = True
        self.thread = threading.Thread(target=self.sender_loop, daemon=True); self.thread.start()

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
        except Exception: return "127.0.0.1"

    def send_ip_udp_broadcast(self):
        def _broadcast():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP); s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self.running:
                try: s.sendto(self.local_ip.encode(), ('<broadcast>', self.udp_port)); time.sleep(1.0) 
                except Exception: pass
            s.close()
        threading.Thread(target=_broadcast, daemon=True).start()

    def send(self, msg):
        with self.lock: self.queue.append(msg)

    def sender_loop(self):
        last_ping = time.time()
        while self.running:
            with self.lock:
                if self.queue:
                    msg_to_send = str(self.queue.pop(0))
                    try: self.socket.send_string(msg_to_send)
                    except Exception: pass
            if time.time() - last_ping > 1.0:
                try: self.socket.send_string("CONNECTED")
                except Exception: pass
                last_ping = time.time()
            time.sleep(0.01) 

    def stop(self):
        self.running = False
        try: self.socket.close(); self.context.term()
        except Exception: pass

# =============================================================================
# JANELA DE CONFIGURAÇÃO (PARADIGMA E CUES)
# =============================================================================
class JanelaConfiguracaoParadigma(QDialog):
    def __init__(self, unity_conectado):
        super().__init__()
        self.setWindowTitle("Configuração de Cues (Visual)")
        self.resize(450, 400); self.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout = QVBoxLayout(self)

        lbl_titulo = QLabel("Configuração de Tempos (Cues)")
        lbl_titulo.setStyleSheet("font-size: 16px; font-weight: bold; color: #00bcd4;"); lbl_titulo.setAlignment(Qt.AlignCenter); layout.addWidget(lbl_titulo)

        group_tempos = QGroupBox("Tempos do Relógio Principal")
        form_tempos = QFormLayout()
        
        self.spin_aviso = QDoubleSpinBox(); self.spin_aviso.setRange(0.5, 5.0); self.spin_aviso.setValue(1.5); self.spin_aviso.setSuffix(" s")
        self.spin_acao = QDoubleSpinBox(); self.spin_acao.setRange(1.0, 10.0); self.spin_acao.setValue(3.0); self.spin_acao.setSuffix(" s")
        self.spin_repouso = QDoubleSpinBox(); self.spin_repouso.setRange(1.0, 10.0); self.spin_repouso.setValue(2.0); self.spin_repouso.setSuffix(" s")
        self.spin_repeticoes = QSpinBox(); self.spin_repeticoes.setRange(1, 100); self.spin_repeticoes.setValue(10); self.spin_repeticoes.setSuffix(" trials/classe")
        
        form_tempos.addRow("Aviso (Cruz ➕):", self.spin_aviso)
        form_tempos.addRow("Ação (Seta ⬅️➡️):", self.spin_acao)
        form_tempos.addRow("Repouso (Preta):", self.spin_repouso)
        form_tempos.addRow("Qtd Repetições:", self.spin_repeticoes)
        group_tempos.setLayout(form_tempos); layout.addWidget(group_tempos)

        group_alvos = QGroupBox("Exibir estímulos em:")
        layout_alvos = QVBoxLayout()
        self.chk_python = QCheckBox("Interface Python (2D)"); self.chk_python.setChecked(True)
        self.chk_unity = QCheckBox("Ambiente Unity (3D)"); self.chk_unity.setChecked(unity_conectado); self.chk_unity.setEnabled(unity_conectado)
        if not unity_conectado: self.chk_unity.setText("Mostrar no Unity (Requer conexão)")
        layout_alvos.addWidget(self.chk_python); layout_alvos.addWidget(self.chk_unity); group_alvos.setLayout(layout_alvos); layout.addWidget(group_alvos)

        layout.addStretch()
        self.btn_iniciar = QPushButton("▶ COMEÇAR PARADIGMA VISUAL")
        self.btn_iniciar.setStyleSheet("background-color: #00bcd4; color: black; font-weight: bold; padding: 12px;")
        self.btn_iniciar.clicked.connect(self.aceitar_configuracao); layout.addWidget(self.btn_iniciar)

    def aceitar_configuracao(self):
        if not self.chk_python.isChecked() and not self.chk_unity.isChecked():
            return QMessageBox.warning(self, "Aviso", "Selecione pelo menos um local para exibir os estímulos!")
        self.configs = {
            't_aviso': int(self.spin_aviso.value() * 1000), 't_acao': int(self.spin_acao.value() * 1000), 't_repouso': int(self.spin_repouso.value() * 1000), 
            'repeticoes': self.spin_repeticoes.value(), 'usar_python': self.chk_python.isChecked(), 'usar_unity': self.chk_unity.isChecked()
        }
        self.accept()

class JanelaExecucaoParadigma(QDialog):
    sessao_concluida = pyqtSignal()

    def __init__(self, configs, unity_sender=None):
        super().__init__()
        self.configs = configs; self.unity = unity_sender
        
        # Gera o Gabarito na hora: Ex [0,0..., 1,1..., 2,2...]
        self.sequencia_trials = [0]*configs['repeticoes'] + [1]*configs['repeticoes'] + [2]*configs['repeticoes']
        random.shuffle(self.sequencia_trials)
        
        self.trial_atual = 0; self.total_trials = len(self.sequencia_trials); self.estado_atual = "INICIO" 
        
        self.setWindowTitle("Coleta Visual (Cues)")
        self.resize(800, 600); self.setStyleSheet("background-color: black; color: white;")
        layout = QVBoxLayout(self)
        
        self.lbl_info = QLabel(f"Preparando sessão... Total: {self.total_trials}"); self.lbl_info.setStyleSheet("font-size: 16px; color: gray;"); self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.lbl_estimulo = QLabel("Pronto?"); self.lbl_estimulo.setAlignment(Qt.AlignCenter); self.lbl_estimulo.setStyleSheet("font-size: 120px; font-weight: bold;")
        layout.addWidget(self.lbl_estimulo, 1)

        self.timer_logica = QTimer(self); self.timer_logica.setSingleShot(True); self.timer_logica.timeout.connect(self.proximo_estado); self.timer_logica.start(2000)

    def desenhar_tela(self, texto, tamanho, cor, info):
        if self.configs['usar_python']:
            self.lbl_estimulo.setText(texto); self.lbl_estimulo.setStyleSheet(f"font-size: {tamanho}px; color: {cor}; font-weight: bold;")
            self.lbl_info.setText(info)
        else:
            self.lbl_estimulo.setText("Transmitindo para o Unity..."); self.lbl_estimulo.setStyleSheet("font-size: 40px; color: #aaaaaa;")
            self.lbl_info.setText(info)

    def proximo_estado(self):
        if self.trial_atual >= self.total_trials:
            if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_REST")
            self.estado_atual = "CONCLUIDO"
            self.desenhar_tela("Concluído!", 80, "#00e676", "Pode fechar esta janela.")
            self.sessao_concluida.emit(); return

        classe_alvo = self.sequencia_trials[self.trial_atual]

        if self.estado_atual == "INICIO" or self.estado_atual == "REPOUSO":
            self.estado_atual = "AVISO"
            if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_CROSS") 
            self.desenhar_tela("➕", 150, "white", f"Estímulo {self.trial_atual + 1}/{self.total_trials} - Foco")
            self.timer_logica.start(self.configs['t_aviso'])
            
        elif self.estado_atual == "AVISO":
            self.estado_atual = "ACAO"
            if classe_alvo == 0:
                if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_LEFT") 
                self.desenhar_tela("⬅️", 180, "#00bcd4", "AÇÃO: Mão Esquerda")
            elif classe_alvo == 1:
                if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_RIGHT") 
                self.desenhar_tela("➡️", 180, "#ff4081", "AÇÃO: Mão Direita")
            elif classe_alvo == 2:
                if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_REST") 
                self.desenhar_tela("🛑", 150, "#ffeb3b", "AÇÃO: Repouso")
            self.timer_logica.start(self.configs['t_acao'])
            
        elif self.estado_atual == "ACAO":
            self.estado_atual = "REPOUSO"
            if self.configs['usar_unity'] and self.unity: self.unity.send("CUE_REST") 
            self.desenhar_tela("", 150, "white", "Descanso...")
            self.trial_atual += 1; self.timer_logica.start(self.configs['t_repouso'])

# =============================================================================
# JANELA PRINCIPAL (MAIN COCKPIT)
# =============================================================================
class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 900); self.setWindowTitle('BCI Control Center Pro')
        self.aplicar_estilo_escuro()

        self.unity = None; self.inlet = None; self.model = None
        self.modo_dados = "TESTE"; self.dados_arquivo = None; self.ponteiro_arquivo = 0
        
        self.canais = ['C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8','T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
        self.n_ch = len(self.canais); self.x_size = 500; self.buffer_sobra = [] 
        
        # Gabarito atual gerado dinamicamente com base no SpinBox
        self.gabarito_sessao = []
        self.total_tentativas = 0; self.indice_atual = 0; self.qtd_tl = 0 
        self.acertos_fase1 = 0; self.acertos_fase2 = 0; self.log_sessao = [] 
        
        self.current_data_visual = np.zeros((self.x_size, self.n_ch)); self.fs = 250.0  
        self.escala_visual = 150; self.escala_auto = False; self.fft_smooth_factor = 0.0
        self.fft_buffer_history = np.zeros((self.n_ch, self.x_size//2))

        self.setup_ui()
        self.setup_menu()

        self.timer_monitor = QTimer(); self.timer_monitor.timeout.connect(self.loop_leitura_dados_continuo); self.timer_monitor.start(20)

    def aplicar_estilo_escuro(self):
        qss = """
        QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', Arial; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 15px; font-weight: bold; background-color: #1e1e1e; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; color: #00bcd4; }
        QPushButton { background-color: #333333; border: 1px solid #555; border-radius: 4px; padding: 8px; color: white; font-weight: bold;}
        QPushButton:hover { background-color: #444444; }
        QPushButton:disabled { background-color: #222222; color: #666666; }
        QTabWidget::pane { border: 1px solid #444; background-color: #1e1e1e; }
        QTabBar::tab { background: #2b2b2b; color: #888888; padding: 8px 25px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: bold; }
        QTabBar::tab:selected { background: #444444; color: #ffffff; border-bottom: 3px solid #00bcd4; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: #333333; border: 1px solid #555; padding: 5px; color: white; border-radius: 3px; }
        QProgressBar { border: 1px solid #555; text-align: center; color: white; border-radius: 3px;}
        QProgressBar::chunk { background-color: #00bcd4; }
        QCheckBox, QRadioButton { color: white; spacing: 5px; font-size: 13px; }
        """
        self.setStyleSheet(qss)

    def setup_ui(self):
        self.centralwidget = QWidget(self); self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        
        self.panel_left = QFrame(); self.panel_left.setFixedWidth(400); self.layout_left = QVBoxLayout(self.panel_left)

        lbl_titulo = QLabel("BCI COCKPIT"); lbl_titulo.setFont(QtGui.QFont("Segoe UI", 16, QtGui.QFont.Bold))
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter); lbl_titulo.setStyleSheet("color: #00bcd4; margin-bottom: 5px;")
        self.layout_left.addWidget(lbl_titulo)

        # --- MODO DE DADOS ---
        group_modo = QGroupBox("1. Seleção de Modo (Fonte de Dados)")
        layout_modo = QVBoxLayout()
        self.rb_online = QRadioButton("📡 ONLINE (LSL Real-time)")
        self.rb_offline = QRadioButton("📁 OFFLINE (Arquivo CSV)")
        self.rb_teste = QRadioButton("🎲 TESTE (Sinais Simulados/Aleatórios)"); self.rb_teste.setChecked(True) 
        
        self.rb_online.toggled.connect(self.atualizar_modo_dados)
        self.rb_offline.toggled.connect(self.atualizar_modo_dados)
        self.rb_teste.toggled.connect(self.atualizar_modo_dados)
        
        layout_modo.addWidget(self.rb_online); layout_modo.addWidget(self.rb_offline); layout_modo.addWidget(self.rb_teste)
        self.lbl_fonte_status = QLabel("Nenhuma fonte conectada."); self.lbl_fonte_status.setStyleSheet("color: gray;")
        self.btn_conectar_fonte = QPushButton("Conectar Fonte de Dados"); self.btn_conectar_fonte.setStyleSheet("background-color: #555555;")
        self.btn_conectar_fonte.clicked.connect(self.conectar_fonte_dados)
        
        layout_modo.addWidget(self.lbl_fonte_status); layout_modo.addWidget(self.btn_conectar_fonte)
        group_modo.setLayout(layout_modo); self.layout_left.addWidget(group_modo)

        # --- UNITY ---
        group_unity = QGroupBox("2. Ambiente Virtual")
        layout_unity = QVBoxLayout()
        self.lbl_status_unity = QLabel("Status: Desconectado"); self.lbl_status_unity.setStyleSheet("color: #ff5555;")
        self.btn_conectar_unity = QPushButton("🎮 Conectar ao Unity (ZMQ)")
        self.btn_conectar_unity.clicked.connect(self.conectar_unity)
        layout_unity.addWidget(self.lbl_status_unity); layout_unity.addWidget(self.btn_conectar_unity)
        group_unity.setLayout(layout_unity); self.layout_left.addWidget(group_unity)

        # --- EXPERIMENTO ---
        group_acoes = QGroupBox("3. Execução do Experimento")
        layout_acoes = QVBoxLayout()
        
        self.btn_paradigma = QPushButton("🎯 PASSO 1: Protocolo de Gravação Visual (Cues)")
        self.btn_paradigma.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; padding: 12px; font-size: 13px;")
        self.btn_paradigma.clicked.connect(self.abrir_gravacao_paradigma)
        layout_acoes.addWidget(self.btn_paradigma)

        # Configurações Dinâmicas de Quantidade e Shape
        form_parametros = QFormLayout()
        
        self.spin_trials_ia = QSpinBox()
        self.spin_trials_ia.setRange(1, 100); self.spin_trials_ia.setValue(10); self.spin_trials_ia.setSuffix(" trials/classe")
        
        self.spin_shape_time = QSpinBox()
        self.spin_shape_time.setRange(10, 5000); self.spin_shape_time.setValue(721); self.spin_shape_time.setSuffix(" pts")
        
        self.spin_shape_ch = QSpinBox()
        self.spin_shape_ch.setRange(1, 32); self.spin_shape_ch.setValue(16); self.spin_shape_ch.setSuffix(" canais")
        
        form_parametros.addRow("Duração da Sessão IA:", self.spin_trials_ia)
        form_parametros.addRow("Shape do Modelo (T):", self.spin_shape_time)
        form_parametros.addRow("Shape do Modelo (C):", self.spin_shape_ch)
        layout_acoes.addLayout(form_parametros)

        # Controle Flexível de Transfer Learning!
        self.combo_tl = QComboBox()
        self.combo_tl.addItems([
            "Somente Avaliação/Teste (0% Treino)", 
            "Treino Contínuo (100% Transfer Learning)", 
            "Misto (20% Treino Inicial -> Teste)"
        ])
        self.combo_tl.setCurrentIndex(2) # Default para Misto
        layout_acoes.addWidget(QLabel("Estratégia de Aprendizado (Transfer Learning):"))
        layout_acoes.addWidget(self.combo_tl)

        self.btn_iniciar_ia = QPushButton("🧠 PASSO 2: Iniciar Sessão Live IA (Hands)")
        self.btn_iniciar_ia.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 12px; font-size: 13px;")
        self.btn_iniciar_ia.clicked.connect(self.iniciar_sessao_ml)
        layout_acoes.addWidget(self.btn_iniciar_ia)

        group_acoes.setLayout(layout_acoes); self.layout_left.addWidget(group_acoes)

        # --- MONITORAMENTO ---
        group_mon = QGroupBox("Monitoramento em Tempo Real")
        layout_mon = QVBoxLayout()
        self.lbl_progresso = QLabel("Progresso: Aguardando..."); self.lbl_progresso.setAlignment(QtCore.Qt.AlignCenter)
        self.bar_progresso = QProgressBar(); self.bar_progresso.setValue(0)
        self.lbl_fase = QLabel("FASE: Parado"); self.lbl_fase.setStyleSheet("color: yellow; font-weight: bold;"); self.lbl_fase.setAlignment(QtCore.Qt.AlignCenter)
        layout_mon.addWidget(self.lbl_progresso); layout_mon.addWidget(self.bar_progresso); layout_mon.addWidget(self.lbl_fase)

        self.lbl_predicao = QLabel("--"); self.lbl_predicao.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold)); self.lbl_predicao.setAlignment(QtCore.Qt.AlignCenter)
        self.gauge = GaugeWidget()
        layout_mon.addWidget(self.lbl_predicao); layout_mon.addWidget(self.gauge)
        group_mon.setLayout(layout_mon); self.layout_left.addWidget(group_mon)

        self.layout_left.addStretch()
        self.main_layout.addWidget(self.panel_left)

        # ==============================================================
        # PAINEL DIREITO (GRÁFICOS)
        # ==============================================================
        self.panel_right = QWidget(); self.layout_right = QVBoxLayout(self.panel_right)
        self.tabs = QTabWidget(); self.setup_abas_graficos(); self.layout_right.addWidget(self.tabs)
        self.main_layout.addWidget(self.panel_right, 1)

    def setup_abas_graficos(self):
        self.tab_time = QWidget(); l_time = QVBoxLayout(self.tab_time)
        tb_time = QHBoxLayout(); self.combo_scale = QComboBox(); self.combo_scale.addItems(["Auto", "50 uV", "100 uV", "200 uV", "400 uV"]); self.combo_scale.setCurrentText("200 uV")
        self.combo_scale.currentTextChanged.connect(lambda t: setattr(self, 'escala_auto', True) if t=="Auto" else (setattr(self, 'escala_auto', False), setattr(self, 'escala_visual', int(t.split()[0])), self.atualizar_limites_temporal()))
        tb_time.addWidget(QLabel("Escala:")); tb_time.addWidget(self.combo_scale); tb_time.addStretch()
        l_time.addLayout(tb_time)
        self.fig_time = Figure(figsize=(5,3), dpi=100, facecolor='#1e1e1e'); self.can_time = FigureCanvas(self.fig_time)
        self.setup_grafico_temporal(); l_time.addWidget(self.can_time); self.tabs.addTab(self.tab_time, "Série Temporal (EEG)")

        self.tab_fft = QWidget(); l_fft = QVBoxLayout(self.tab_fft)
        tb_fft = QHBoxLayout(); self.spin_smooth = QDoubleSpinBox(); self.spin_smooth.setRange(0, 0.99); self.spin_smooth.setSingleStep(0.1)
        self.spin_smooth.valueChanged.connect(lambda: setattr(self, 'fft_smooth_factor', self.spin_smooth.value()))
        tb_fft.addWidget(QLabel("Smooth:")); tb_fft.addWidget(self.spin_smooth); tb_fft.addStretch()
        l_fft.addLayout(tb_fft)
        self.fig_fft = Figure(figsize=(5,3), dpi=100, facecolor='#1e1e1e'); self.can_fft = FigureCanvas(self.fig_fft)
        self.setup_grafico_fft(); l_fft.addWidget(self.can_fft); self.tabs.addTab(self.tab_fft, "Domínio da Frequência (FFT)")

    def setup_grafico_temporal(self):
        self.ax_time = self.fig_time.add_subplot(111); self.ax_time.set_facecolor('#1e1e1e'); self.ax_time.tick_params(colors='#aaaaaa')
        self.ax_time.set_xlim(0, self.x_size); self.ax_time.set_yticks([])
        for spine in self.ax_time.spines.values(): spine.set_color('#444444')
        colors = ['#00bcd4', '#ff4081', '#71c671', '#e8c346', '#e68136', '#8959a8', '#d84e4e', '#8c564b']; self.lines_time = []; self.rms_texts = []
        for i in range(self.n_ch):
            l, = self.ax_time.plot([],[], lw=1.2, color=colors[i%8])
            self.lines_time.append(l)
            self.rms_texts.append(self.ax_time.text(self.x_size+5, 0, "", fontsize=8, color='#aaaaaa'))
        self.atualizar_limites_temporal()

    def setup_grafico_fft(self):
        self.ax_fft = self.fig_fft.add_subplot(111); self.ax_fft.set_facecolor('#1e1e1e'); self.ax_fft.tick_params(colors='#aaaaaa', which='both')
        self.ax_fft.set_yscale('log'); self.ax_fft.set_ylim(0.1, 100); self.ax_fft.set_xlim(0, 60); self.ax_fft.grid(True, which='both', color='#333333', alpha=0.8)
        self.ax_fft.set_xlabel('Freq (Hz)', color='#aaaaaa'); self.ax_fft.set_ylabel('uV', color='#aaaaaa')
        for spine in self.ax_fft.spines.values(): spine.set_color('#444444')
        colors = ['#00bcd4', '#ff4081', '#71c671', '#e8c346', '#e68136', '#8959a8', '#d84e4e', '#8c564b']
        self.lines_fft = [self.ax_fft.plot([],[], lw=1.5, alpha=0.8, color=colors[i%8])[0] for i in range(self.n_ch)]

    def atualizar_limites_temporal(self):
        top = self.n_ch * self.escala_visual; self.ax_time.set_ylim(-self.escala_visual, top + self.escala_visual)

    def setup_menu(self):
        menu_arquivo = self.menuBar().addMenu("Arquivo")
        menu_arquivo.addAction("Carregar Modelo de IA (.h5)").triggered.connect(self.carregar_modelo_arquivo)
        menu_arquivo.addAction("Salvar Modelo Fine-Tuned (.h5)").triggered.connect(self.salvar_modelo_arquivo)

    # =========================================================================
    # FUNÇÕES LÓGICAS E DE CONEXÃO
    # =========================================================================
    def atualizar_modo_dados(self):
        if self.rb_online.isChecked():
            self.modo_dados = "ONLINE"; self.btn_conectar_fonte.setText("Procurar Placa LSL na Rede"); self.lbl_fonte_status.setText("Aguardando busca LSL...")
        elif self.rb_offline.isChecked():
            self.modo_dados = "OFFLINE"; self.btn_conectar_fonte.setText("Procurar Arquivo CSV local"); self.lbl_fonte_status.setText("Aguardando arquivo CSV...")
        else:
            self.modo_dados = "TESTE"; self.btn_conectar_fonte.setText("Modo Simulação Ativo"); self.lbl_fonte_status.setText("Fonte: Gerador Aleatório de Numpy"); self.lbl_fonte_status.setStyleSheet("color: #00e676;")

    def conectar_fonte_dados(self):
        if self.modo_dados == "ONLINE":
            self.lbl_fonte_status.setText("Buscando streams LSL..."); QApplication.processEvents()
            streams = resolve_byprop('type', 'EEG', timeout=3)
            if streams:
                self.inlet = StreamInlet(streams[0]); self.lbl_fonte_status.setText(f"LSL Conectado: {streams[0].name()}"); self.lbl_fonte_status.setStyleSheet("color: #00e676;")
                QMessageBox.information(self, "Sucesso", "Placa LSL capturada!")
            else:
                self.lbl_fonte_status.setText("Falha: Nenhum LSL encontrado."); QMessageBox.warning(self, "Erro", "Não foi possível achar uma placa transmitindo dados LSL.")
        elif self.modo_dados == "OFFLINE":
            fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Gravação Offline', '', "Arquivos CSV (*.csv)")
            if fname:
                try:
                    df = pd.read_csv(fname, comment='%'); self.dados_arquivo = df.iloc[:, 1 : self.n_ch + 1].values; self.ponteiro_arquivo = 0
                    self.lbl_fonte_status.setText(f"CSV Ativo: {fname.split('/')[-1]} ({len(self.dados_arquivo)} linhas)"); self.lbl_fonte_status.setStyleSheet("color: #00e676;")
                except Exception as e: QMessageBox.critical(self, "Erro", f"O arquivo CSV está corrompido ou é inválido.\nDetalhe: {e}")

    def conectar_unity(self):
        if not self.unity:
            try:
                self.unity = UnitySender(); self.lbl_status_unity.setText("Status: Conectado (ZMQ Porta 5555)"); self.lbl_status_unity.setStyleSheet("color: #00e676;"); self.btn_conectar_unity.setEnabled(False)
            except Exception as e: QMessageBox.critical(self, "Erro de Rede", f"Não foi possível criar servidor ZMQ.\n{e}")

    def carregar_modelo_arquivo(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir Modelo IA', '', "Keras Model (*.h5)")
        if fname and USAR_MODELO:
            try:
                old = load_model(fname)
                if old.output_shape[-1] != 3:
                    new = Sequential(); new.add(Input(shape=old.input_shape[1:]))
                    for l in old.layers[:-1]: new.add(l)
                    new.add(Dense(3, activation='softmax')); self.model = new
                else: self.model = old
                self.model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                QMessageBox.information(self, "Sucesso", f"Modelo '{fname.split('/')[-1]}' carregado e pronto.")
            except Exception as e: QMessageBox.critical(self, "Erro Keras", f"Falha ao carregar modelo:\n{e}")

    def salvar_modelo_arquivo(self):
        if not self.model: return QMessageBox.warning(self, "Aviso", "Nenhum modelo carregado na memória para salvar.")
        fname, _ = QFileDialog.getSaveFileName(self, 'Salvar Novo Modelo Fine-Tuned', "modelo_TL_atualizado.h5", "Keras Model (*.h5)")
        if fname:
            try: self.model.save(fname); QMessageBox.information(self, "Sucesso", "Modelo exportado com sucesso!")
            except Exception as e: QMessageBox.critical(self, "Erro", f"Erro ao tentar salvar arquivo:\n{e}")

    # =========================================================================
    # CONTROLE DE FLUXO E LEITURA (BACKGROUND)
    # =========================================================================
    def abrir_gravacao_paradigma(self):
        win_config = JanelaConfiguracaoParadigma(self.unity is not None)
        if win_config.exec_() == QDialog.Accepted:
            configs = win_config.configs
            self.paradigma_win = JanelaExecucaoParadigma(configs, unity_sender=self.unity)
            self.paradigma_win.sessao_concluida.connect(self.receber_gabarito_da_gravacao)
            self.paradigma_win.show() 

    def receber_gabarito_da_gravacao(self):
        QMessageBox.information(self, "Coleta Concluída", "Protocolo visual encerrado.")

    def loop_leitura_dados_continuo(self):
        teve_dado_novo = False
        if self.modo_dados == "TESTE":
            data = np.random.randn(3, self.n_ch) * 50 
            self.buffer_sobra.extend(data); self.current_data_visual = np.roll(self.current_data_visual, -3, axis=0); self.current_data_visual[-3:, :] = data
            teve_dado_novo = True
        elif self.modo_dados == "OFFLINE" and self.dados_arquivo is not None:
            chunk_size = 3 
            if self.ponteiro_arquivo + chunk_size < len(self.dados_arquivo):
                data = self.dados_arquivo[self.ponteiro_arquivo : self.ponteiro_arquivo + chunk_size]
                self.ponteiro_arquivo += chunk_size; self.buffer_sobra.extend(data)
                self.current_data_visual = np.roll(self.current_data_visual, -len(data), axis=0); self.current_data_visual[-len(data):, :] = data
                teve_dado_novo = True
            else: self.ponteiro_arquivo = 0 
        elif self.modo_dados == "ONLINE" and self.inlet:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0)
            if chunk:
                data = np.array(chunk)[:, :self.n_ch]; self.buffer_sobra.extend(data)
                self.current_data_visual = np.roll(self.current_data_visual, -len(data), axis=0); self.current_data_visual[-len(data):, :] = data
                teve_dado_novo = True

        if teve_dado_novo: self.atualizar_graficos_visuais()

    def atualizar_graficos_visuais(self):
        if self.tabs.currentIndex() == 0: 
            if self.escala_auto:
                amp = np.ptp(self.current_data_visual, axis=0).max()
                if amp > 1: self.escala_visual = amp * 0.8; self.atualizar_limites_temporal()
            x = np.arange(self.x_size)
            for i, l in enumerate(self.lines_time):
                off = i * self.escala_visual; y = self.current_data_visual[:, i] - np.mean(self.current_data_visual[:, i])
                l.set_data(x, y + off); rms = np.sqrt(np.mean(y**2)); self.rms_texts[i].set_text(f"{rms:.1f} uV"); self.rms_texts[i].set_position((self.x_size+5, off))
            self.can_time.draw_idle()
        elif self.tabs.currentIndex() == 1: 
            xf = np.linspace(0, self.fs/2, self.x_size//2)
            for i, l in enumerate(self.lines_fft):
                raw = 2.0/self.x_size * np.abs(fft(self.current_data_visual[:, i])[0:self.x_size//2])
                f = self.fft_smooth_factor; self.fft_buffer_history[i] = (self.fft_buffer_history[i]*f) + (raw*(1-f))
                l.set_data(xf, self.fft_buffer_history[i])
            self.can_fft.draw_idle()

    # =========================================================================
    # MOTOR DE IA E TRANSFER LEARNING
    # =========================================================================
    def iniciar_sessao_ml(self):
        if self.modo_dados == "ONLINE" and not self.inlet:
            return QMessageBox.warning(self, "Aviso", "Conecte o LSL primeiro.")
        if self.modo_dados == "OFFLINE" and self.dados_arquivo is None:
            return QMessageBox.warning(self, "Aviso", "Abra um arquivo CSV primeiro.")

        # Cria o Gabarito com base no SpinBox e embaralha para a sessão da IA
        rep_por_classe = self.spin_trials_ia.value()
        self.gabarito_sessao = [0]*rep_por_classe + [1]*rep_por_classe + [2]*rep_por_classe
        random.shuffle(self.gabarito_sessao)
        self.total_tentativas = len(self.gabarito_sessao)

        # Lógica de Controle do Transfer Learning
        estrategia_tl = self.combo_tl.currentText()
        if "Misto" in estrategia_tl:
            self.qtd_tl = int(self.total_tentativas * 0.2) # Usa 20%
        elif "Treino Contínuo" in estrategia_tl:
            self.qtd_tl = self.total_tentativas # Treina em todos
        else:
            self.qtd_tl = 0 # Nunca treina
            
        self.indice_atual = 0; self.acertos_fase1 = 0; self.acertos_fase2 = 0; self.log_sessao = []
        
        self.btn_iniciar_ia.setEnabled(False); self.btn_iniciar_ia.setText("Sessão Live Rodando...")
        self.bar_progresso.setMaximum(self.total_tentativas); self.bar_progresso.setValue(0)
        
        self.timer_sessao = QtCore.QTimer(); self.timer_sessao.timeout.connect(self.loop_sessao_ml); self.timer_sessao.start(10)

    def loop_sessao_ml(self):
        target_time = self.spin_shape_time.value()
        target_ch = self.spin_shape_ch.value()
        
        if len(self.buffer_sobra) >= target_time:
            if self.indice_atual >= self.total_tentativas: 
                self.finalizar_sessao()
                return
            
            raw_epoch = np.array(self.buffer_sobra[:target_time])
            self.buffer_sobra = self.buffer_sobra[target_time:] 
            dados_para_ia = raw_epoch[:, :target_ch]
            self.classificar_e_treinar(dados_para_ia)

    def classificar_e_treinar(self, dados):
        lbl_real = self.gabarito_sessao[self.indice_atual]
        pred = 2; prob = [0.0, 0.0, 1.0]

        if self.modo_dados == "TESTE" or not self.model:
            pred = random.randint(0, 2); prob = [1.0 if i==pred else 0.0 for i in range(3)]
        else:
            dados_norm = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8)
            input_data = np.expand_dims(dados_norm, axis=0).astype(np.float32)
            try:
                res = self.model.predict(input_data, verbose=0)[0]
                pred = np.argmax(res); prob = res
            except Exception as e: print(f"Erro na predição: {e}")

        # Salva o Log no Dicionário
        self.log_sessao.append({
            'Tentativa': self.indice_atual + 1, 'Timestamp': datetime.now().strftime('%H:%M:%S.%f'),
            'Label_Verdadeiro': lbl_real, 'Predicao_IA': pred,
            'Prob_Esq': round(prob[0], 4), 'Prob_Dir': round(prob[1], 4), 'Prob_Rep': round(prob[2], 4)
        })

        fase_nome = "TREINAMENTO (TL)" if self.indice_atual < self.qtd_tl else "AVALIAÇÃO DE DESEMPENHO"
        self.lbl_fase.setText(f"FASE: {fase_nome}")
        self.lbl_fase.setStyleSheet(f"color: {'yellow' if self.indice_atual < self.qtd_tl else '#00e676'}; font-weight: bold;")
        self.lbl_progresso.setText(f"Progresso: Época {self.indice_atual+1} / {self.total_tentativas}")
        self.bar_progresso.setValue(self.indice_atual + 1)
        
        nomes = ["MÃO ESQUERDA", "MÃO DIREITA", "REPOUSO"]; cores = ["#00bcd4", "#ff4081", "#ffffff"]
        self.lbl_predicao.setText(nomes[pred]); self.lbl_predicao.setStyleSheet(f"color: {cores[pred]}")
        self.gauge.set_probabilities(prob[0], prob[1], prob[2])

        acertou = (pred == lbl_real)

        # Envia comando de MOVIMENTO para o Unity
        if self.unity:
            if pred == 0: self.unity.send("HAND_LEFT")
            elif pred == 1: self.unity.send("HAND_RIGHT")
            else: self.unity.send("HAND_REST")

        # FINE TUNING (O Ouro da Pesquisa)
        if self.modo_dados != "TESTE" and self.model and self.indice_atual < self.qtd_tl:
            if acertou: 
                self.acertos_fase1 += 1
                d_norm = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8)
                inp = np.expand_dims(d_norm, axis=0).astype(np.float32)
                target = np.array([lbl_real]).astype(np.float32)
                for _ in range(EPOCHS_TREINO): self.model.train_on_batch(inp, target)
        elif self.modo_dados != "TESTE" and acertou:
            self.acertos_fase2 += 1

        self.indice_atual += 1

    def finalizar_sessao(self):
        self.timer_sessao.stop()
        self.btn_iniciar_ia.setEnabled(True); self.btn_iniciar_ia.setText("🧠 PASSO 2: Iniciar Sessão Live IA (Hands)")
        if self.unity: self.unity.send("HAND_REST")

        mensagem_final = "A sessão terminou com sucesso!"
        if len(self.log_sessao) > 0:
            try:
                df_log = pd.DataFrame(self.log_sessao)
                nome_csv = f"bci_sessao_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_log.to_csv(nome_csv, index=False)
                mensagem_final += f"\n\nOs resultados foram salvos na pasta raiz em:\n{nome_csv}"
            except Exception as e: mensagem_final += f"\n\nATENÇÃO: Falha ao salvar arquivo CSV. {e}"

        if self.modo_dados != "TESTE":
            total_teste = self.total_tentativas - self.qtd_tl
            acc_treino = (self.acertos_fase1 / self.qtd_tl) * 100 if self.qtd_tl > 0 else 0
            acc_teste = (self.acertos_fase2 / total_teste) * 100 if total_teste > 0 else 0
            mensagem_final += f"\n\nEstatísticas da IA:\nAcertos no Treino (TL): {acc_treino:.1f}%\nAcertos no Teste Real: {acc_teste:.1f}%"

        QMessageBox.information(self, "Fim de Sessão", mensagem_final)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = JanelaInicial()
    win.show()
    sys.exit(app.exec_())