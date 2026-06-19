from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QPolygon
from PyQt5.QtCore import Qt, QPoint, QRectF, QTimer
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

try:
    from config import GABARITO_SESSAO, PORCENTAGEM_TL, EPOCHS_TREINO
except ImportError:
    GABARITO_SESSAO = [0, 1, 2] * 20 
    PORCENTAGEM_TL = 0.2
    EPOCHS_TREINO = 1

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

class GaugeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(250, 130)
        self.current_angle = 0.0
        self.base_angle = 0.0   
        self.target_angle = 0.0
        self.incerteza = 0.0    
        
        self.anim_timer = QtCore.QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(30) 

    def set_probabilities(self, prob_left, prob_right, prob_rest):
        self.target_angle = (prob_right * 60) + (prob_left * -60)
        confianca_maxima = max(prob_left, prob_right, prob_rest)
        self.incerteza = 1.0 - confianca_maxima

    def update_animation(self):
        diff = self.target_angle - self.base_angle
        if abs(diff) > 0.1: self.base_angle += diff * 0.15
        tremor_atual = random.uniform(-1.0, 1.0) * (self.incerteza * 15.0)
        self.current_angle = self.base_angle + tremor_atual
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h - 20
        radius = min(w / 2, h) - 25
        rect = QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        painter.setPen(QPen(QColor("#00bcd4"), 15, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(rect, 120 * 16, 60 * 16) 
        painter.setPen(QPen(QColor("#888888"), 15, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(rect, 60 * 16, 60 * 16)
        painter.setPen(QPen(QColor("#ff4081"), 15, Qt.SolidLine, Qt.FlatCap))
        painter.drawArc(rect, 0 * 16, 60 * 16)

        painter.setPen(QColor("#ffffff"))
        painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        painter.drawText(int(center_x - radius - 20), int(center_y), "ESQ")
        painter.drawText(int(center_x + radius + 0), int(center_y), "DIR")
        painter.drawText(int(center_x - 15), int(center_y - radius - 15), "REP")

        painter.translate(center_x, center_y)
        painter.rotate(self.current_angle)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#ffffff")))
        poly = QPolygon([QPoint(-4, 0), QPoint(4, 0), QPoint(0, int(-radius + 5))])
        painter.drawPolygon(poly)
        painter.setBrush(QBrush(QColor("#ffc107")))
        painter.drawEllipse(QPoint(0, 0), 6, 6)

class UnitySender:
    def __init__(self, port=PORTA_UNITY, udp_port=PORTA_UDP_UNITY):
        self.port = port
        self.udp_port = udp_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        try: self.socket.setsockopt(zmq.CONFLATE, 1)
        except: pass 
        try: self.socket.bind(f"tcp://*:{port}")
        except: pass 

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
                try: s.sendto(self.local_ip.encode(), ('<broadcast>', self.udp_port)); time.sleep(1.0) 
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

class JanelaConfiguracaoParadigma(QDialog):
    def __init__(self, unity_conectado):
        super().__init__()
        self.unity_conectado = unity_conectado
        self.setWindowTitle("Configuração do Protocolo BCI")
        self.resize(550, 500)
        self.aplicar_estilo()
        layout = QVBoxLayout(self)

        lbl_titulo = QLabel("Gravação de Paradigma (Master Clock)")
        lbl_titulo.setStyleSheet("font-size: 16px; font-weight: bold; color: #00bcd4; margin-bottom: 10px;")
        lbl_titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_titulo)

        # 1. Classes
        group_classes = QGroupBox("1. Número de Classes")
        layout_classes = QVBoxLayout()
        self.radio_2_classes = QRadioButton("2 Classes (Esq/Dir)")
        self.radio_3_classes = QRadioButton("3 Classes (Esq/Dir + Repouso)")
        self.radio_3_classes.setChecked(True)
        layout_classes.addWidget(self.radio_2_classes); layout_classes.addWidget(self.radio_3_classes)
        group_classes.setLayout(layout_classes); layout.addWidget(group_classes)

        # 2. Tempos
        group_tempos = QGroupBox("2. Tempos no Relógio Principal")
        form_tempos = QFormLayout()
        self.spin_aviso = QDoubleSpinBox(); self.spin_aviso.setRange(0.5, 5.0); self.spin_aviso.setValue(1.5); self.spin_aviso.setSuffix(" s")
        self.spin_acao = QDoubleSpinBox(); self.spin_acao.setRange(1.0, 10.0); self.spin_acao.setValue(3.0); self.spin_acao.setSuffix(" s")
        self.spin_repouso = QDoubleSpinBox(); self.spin_repouso.setRange(1.0, 10.0); self.spin_repouso.setValue(2.0); self.spin_repouso.setSuffix(" s")
        self.spin_trials = QSpinBox(); self.spin_trials.setRange(1, 100); self.spin_trials.setValue(15)
        form_tempos.addRow("Aviso (Cruz de Foco ➕):", self.spin_aviso)
        form_tempos.addRow("Execução (Seta/Stop):", self.spin_acao)
        form_tempos.addRow("Descanso (Tela Preta):", self.spin_repouso)
        form_tempos.addRow("Repetições:", self.spin_trials)
        group_tempos.setLayout(form_tempos); layout.addWidget(group_tempos)

        # 3. CONTROLE DE ONDE EXIBIR (NOVIDADE)
        group_alvos = QGroupBox("3. Onde exibir os estímulos?")
        layout_alvos = QVBoxLayout()
        self.chk_python = QCheckBox("Interface Python (Monitor 2D)")
        self.chk_python.setChecked(True)
        self.chk_unity = QCheckBox("Realidade Virtual (Unity 3D)")
        self.chk_unity.setChecked(self.unity_conectado)
        self.chk_unity.setEnabled(self.unity_conectado)
        if not self.unity_conectado:
            self.chk_unity.setText("Realidade Virtual (Unity não conectado)")
            
        layout_alvos.addWidget(self.chk_python)
        layout_alvos.addWidget(self.chk_unity)
        group_alvos.setLayout(layout_alvos); layout.addWidget(group_alvos)

        self.btn_iniciar = QPushButton("▶ INICIAR CONTROLE DO PARADIGMA")
        self.btn_iniciar.setStyleSheet("background-color: #2e7d32; font-size: 14px; padding: 10px; font-weight: bold;")
        self.btn_iniciar.clicked.connect(self.aceitar_configuracao)
        layout.addStretch(); layout.addWidget(self.btn_iniciar)

    def aplicar_estilo(self):
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI'; }
            QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #00bcd4; }
            QPushButton { background-color: #3c3f41; border: 1px solid #555; border-radius: 4px; padding: 6px; color: white; }
            QDoubleSpinBox, QSpinBox, QRadioButton, QCheckBox { background: #3c3f41; color: white; border: 1px solid #555; padding: 3px; }
        """)

    def aceitar_configuracao(self):
        if not self.chk_python.isChecked() and not self.chk_unity.isChecked():
            QMessageBox.warning(self, "Aviso", "Você precisa selecionar pelo menos um local para exibir os estímulos!")
            return
            
        self.configs = {
            '3_classes': self.radio_3_classes.isChecked(),
            't_aviso': int(self.spin_aviso.value() * 1000),     
            't_acao': int(self.spin_acao.value() * 1000),       
            't_repouso': int(self.spin_repouso.value() * 1000), 
            'trials_por_classe': self.spin_trials.value(),
            'usar_python': self.chk_python.isChecked(),
            'usar_unity': self.chk_unity.isChecked()
        }
        self.accept()

# =============================================================================
# JANELA DE EXECUÇÃO: CONTROLA PYTHON E/OU UNITY
# =============================================================================
class JanelaExecucaoParadigma(QDialog):
    def __init__(self, configs, unity_sender=None):
        super().__init__()
        self.configs = configs
        self.unity = unity_sender 
        
        self.setWindowTitle("Sessão de Gravação BCI")
        self.resize(800, 600)
        self.setStyleSheet("background-color: black; color: white;")
        
        if configs['3_classes']:
            acoes_ativas = [0] * configs['trials_por_classe'] + [1] * configs['trials_por_classe']
            random.shuffle(acoes_ativas)
            self.sequencia_trials = []
            for acao in acoes_ativas:
                self.sequencia_trials.append(acao)
                self.sequencia_trials.append(2) 
        else:
            self.sequencia_trials = [0] * configs['trials_por_classe'] + [1] * configs['trials_por_classe']
            random.shuffle(self.sequencia_trials)
            
        self.trial_atual = 0
        self.total_trials = len(self.sequencia_trials)
        self.estado_atual = "INICIO" 
        
        self.tempo_alvo = 0.0
        self.timer_ui = QTimer(self)
        self.timer_ui.timeout.connect(self.atualizar_relogio)
        self.timer_ui.start(50) 

        layout = QVBoxLayout(self)
        layout_topo = QHBoxLayout()
        self.lbl_info = QLabel(f"Pressione o botão para iniciar. Total: {self.total_trials}")
        self.lbl_info.setStyleSheet("font-size: 16px; color: gray;")
        
        self.lbl_cronometro = QLabel("0.0 s")
        self.lbl_cronometro.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_cronometro.setStyleSheet("font-size: 24px; color: #ff5555; font-weight: bold;")
        
        layout_topo.addWidget(self.lbl_info); layout_topo.addWidget(self.lbl_cronometro)
        layout.addLayout(layout_topo)

        self.lbl_estimulo = QLabel("Pronto?")
        self.lbl_estimulo.setAlignment(Qt.AlignCenter)
        self.lbl_estimulo.setStyleSheet("font-size: 120px; font-weight: bold;")
        layout.addWidget(self.lbl_estimulo, 1)

        self.btn_comecar = QPushButton("Começar!")
        self.btn_comecar.setStyleSheet("background-color: #00bcd4; color: black; font-size: 16px; padding: 10px; font-weight: bold;")
        self.btn_comecar.clicked.connect(self.iniciar_ciclo)
        layout.addWidget(self.btn_comecar)

        self.timer_logica = QTimer(self)
        self.timer_logica.setSingleShot(True)
        self.timer_logica.timeout.connect(self.proximo_estado)

    def iniciar_ciclo(self):
        self.btn_comecar.hide()
        self.proximo_estado()
        
    def atualizar_relogio(self):
        if self.estado_atual in ["INICIO", "CONCLUIDO"]: return
        tempo_restante = max(0.0, self.tempo_alvo - time.time())
        self.lbl_cronometro.setText(f"{tempo_restante:.1f} s")

    def desenhar_tela_python(self, texto, tamanho, cor, info):
        if self.configs['usar_python']:
            self.lbl_estimulo.setText(texto)
            self.lbl_estimulo.setStyleSheet(f"font-size: {tamanho}px; color: {cor}; font-weight: bold;")
            self.lbl_info.setText(info)
        else:
            self.lbl_estimulo.setText("Transmitindo para o VR...")
            self.lbl_estimulo.setStyleSheet("font-size: 40px; color: #aaaaaa;")
            self.lbl_info.setText(info)

    def proximo_estado(self):
        if self.trial_atual >= self.total_trials:
            if self.configs['usar_unity'] and self.unity: self.unity.send("SHOW_REST")
            self.estado_atual = "CONCLUIDO"
            self.desenhar_tela_python("Sessão\nConcluída!", 60, "#00e676", "Feche esta janela.")
            self.lbl_cronometro.setText("0.0 s")
            return

        if self.estado_atual == "INICIO" or self.estado_atual == "REPOUSO":
            self.estado_atual = "AVISO"
            
            # Manda pro Unity
            if self.configs['usar_unity'] and self.unity: self.unity.send("SHOW_REST") 
            
            classe_alvo = self.sequencia_trials[self.trial_atual]
            tipo_acao = "Repouso" if classe_alvo == 2 else "Movimento"
            info = f"Estímulo {self.trial_atual + 1}/{self.total_trials} ({tipo_acao}) - Prepare-se..."
            
            # Desenha no Python
            self.desenhar_tela_python("➕", 150, "white", info)
            
            duracao = self.configs['t_aviso']
            self.tempo_alvo = time.time() + (duracao / 1000.0)
            self.timer_logica.start(duracao)
            
        elif self.estado_atual == "AVISO":
            self.estado_atual = "ACAO"
            classe_alvo = self.sequencia_trials[self.trial_atual]
            
            if classe_alvo == 0:
                if self.configs['usar_unity'] and self.unity: self.unity.send("SHOW_LEFT") 
                self.desenhar_tela_python("⬅️", 180, "#00bcd4", "Execute o Movimento: MÃO ESQUERDA")
                
            elif classe_alvo == 1:
                if self.configs['usar_unity'] and self.unity: self.unity.send("SHOW_RIGHT") 
                self.desenhar_tela_python("➡️", 180, "#ff4081", "Execute o Movimento: MÃO DIREITA")
                
            elif classe_alvo == 2:
                if self.configs['usar_unity'] and self.unity: self.unity.send("SHOW_REST") 
                self.desenhar_tela_python("🛑", 150, "#ffeb3b", "RELAXE e foque no sinal")

            duracao = self.configs['t_acao']
            self.tempo_alvo = time.time() + (duracao / 1000.0)
            self.timer_logica.start(duracao)
            
        elif self.estado_atual == "ACAO":
            self.estado_atual = "REPOUSO"
            if self.configs['usar_unity'] and self.unity: self.unity.send("SHOW_REST") 
            
            self.desenhar_tela_python("", 150, "white", "Tela de Descanso...")
            self.trial_atual += 1
            
            duracao = self.configs['t_repouso']
            self.tempo_alvo = time.time() + (duracao / 1000.0)
            self.timer_logica.start(duracao)

# =============================================================================
# JANELA INICIAL
# =============================================================================
class JanelaInicial(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1300, 900) 
        self.setWindowTitle('BCI Control Center')
        self.aplicar_estilo_escuro()

        self.unity = None
        self.inlet = None
        self.model = None
        self.conectado_unity = False
        self.sessao_iniciada = False
        self.sincronizado = False
        self.modo_teste_unity = False 
        self.modo_arquivo = False
        self.dados_arquivo = None
        self.ponteiro_arquivo = 0
        
        self.canais = ['C3', 'C4', 'Fp1', 'Fp2', 'F7', 'F3', 'F4', 'F8','T7', 'T8', 'P7', 'P3', 'P4', 'P8', 'O1', 'O2']
        self.n_channels_hardware = len(self.canais) 
        self.x_size = 500 
        self.buffer_sobra = [] 
        
        self.indice_atual = 0
        self.gabarito_sessao = GABARITO_SESSAO 
        self.total_tentativas = len(self.gabarito_sessao)
        self.qtd_tl = 0 
        self.acertos_fase1 = 0
        self.acertos_fase2 = 0
        
        self.current_data_visual = np.zeros((self.x_size, self.n_channels_hardware))
        self.fs = 250.0  
        self.escala_visual = 150 
        self.escala_auto = False
        self.fft_smooth_factor = 0.0
        self.fft_buffer_history = np.zeros((self.n_channels_hardware, self.x_size//2))

        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QHBoxLayout(self.centralwidget)
        
        self.panel_left = QFrame()
        self.panel_left.setFixedWidth(360) 
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
        lbl_titulo = QLabel("BCI CONTROL CENTER")
        lbl_titulo.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        lbl_titulo.setAlignment(QtCore.Qt.AlignCenter)
        lbl_titulo.setStyleSheet("color: #00bcd4; margin-bottom: 5px;")
        self.layout_left.addWidget(lbl_titulo)

        group_conn = QGroupBox("1. Fontes de Dados e Conexões")
        layout_conn = QVBoxLayout()
        form_conn = QFormLayout()
        self.lbl_lsl = QLabel("Desconectado"); self.lbl_lsl.setStyleSheet("color: #ff5555;")
        self.lbl_csv = QLabel("Nenhum"); self.lbl_csv.setStyleSheet("color: gray;")
        self.lbl_unity = QLabel("Desconectado"); self.lbl_unity.setStyleSheet("color: #ff5555;")
        self.lbl_model = QLabel("Nenhum"); self.lbl_model.setStyleSheet("color: gray;")
        form_conn.addRow("LSL (Tempo Real):", self.lbl_lsl)
        form_conn.addRow("CSV (Simulação):", self.lbl_csv)
        form_conn.addRow("Unity:", self.lbl_unity)
        form_conn.addRow("IA Base:", self.lbl_model)
        layout_conn.addLayout(form_conn)

        self.btn_csv = QPushButton("📁 Carregar Arquivo CSV"); self.btn_csv.clicked.connect(self.carregar_csv)
        self.btn_lsl = QPushButton("📡 Conectar Placa (LSL)"); self.btn_lsl.clicked.connect(self.conectar_LSL)
        self.btn_unity = QPushButton("🎮 Conectar ao Unity"); self.btn_unity.clicked.connect(self.conectarUnity)
        
        row_btns = QHBoxLayout()
        row_btns.addWidget(self.btn_csv); row_btns.addWidget(self.btn_lsl)
        layout_conn.addLayout(row_btns); layout_conn.addWidget(self.btn_unity)
        group_conn.setLayout(layout_conn); self.layout_left.addWidget(group_conn)

        group_config = QGroupBox("2. Configuração de Treino")
        layout_config = QVBoxLayout()
        layout_config.addWidget(QLabel("Modo da IA na Sessão:"))
        self.radio_com_tl = QRadioButton(f"Com Transfer Learning ({int(PORCENTAGEM_TL*100)}% da base)")
        self.radio_sem_tl = QRadioButton("Sem Transfer Learning (Apenas Teste)")
        self.radio_com_tl.setChecked(True) 
        layout_config.addWidget(self.radio_com_tl); layout_config.addWidget(self.radio_sem_tl)

        self.chk_teste_unity = QCheckBox("Modo Teste Unity (Gerar sinais aleatórios)")
        self.chk_teste_unity.setStyleSheet("color: #ff9800; font-weight: bold; margin-top: 5px;")
        layout_config.addWidget(self.chk_teste_unity)
        group_config.setLayout(layout_config); self.layout_left.addWidget(group_config)

        group_prog = QGroupBox("3. Execução")
        layout_prog = QVBoxLayout()
        self.btn_iniciar = QPushButton("▶ INICIAR SESSÃO")
        self.btn_iniciar.setStyleSheet("background-color: #2e7d32; font-size: 14px; padding: 10px; font-weight: bold;")
        self.btn_iniciar.clicked.connect(self.iniciar_sessao)
        layout_prog.addWidget(self.btn_iniciar)

        self.lbl_progresso = QLabel("Tentativa: 0 / 0"); self.lbl_progresso.setAlignment(QtCore.Qt.AlignCenter)
        self.bar_progresso = QProgressBar(); self.bar_progresso.setValue(0)
        self.lbl_fase = QLabel("AGUARDANDO INÍCIO"); self.lbl_fase.setStyleSheet("color: yellow; font-weight: bold;"); self.lbl_fase.setAlignment(QtCore.Qt.AlignCenter)
        
        layout_prog.addWidget(self.lbl_progresso); layout_prog.addWidget(self.bar_progresso); layout_prog.addWidget(self.lbl_fase)
        group_prog.setLayout(layout_prog); self.layout_left.addWidget(group_prog)

        group_res = QGroupBox("4. Predição e Pureza de Sinal")
        layout_res = QVBoxLayout()
        self.lbl_predicao = QLabel("--")
        self.lbl_predicao.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        self.lbl_predicao.setAlignment(QtCore.Qt.AlignCenter)
        
        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_feedback.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))

        self.gauge = GaugeWidget()
        layout_res.addWidget(self.lbl_predicao); layout_res.addWidget(self.lbl_feedback); layout_res.addWidget(self.gauge)
        
        group_res.setLayout(layout_res); self.layout_left.addWidget(group_res)
        self.layout_left.addStretch()

    def aplicar_estilo_escuro(self):
        qss = """
        QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', Arial; }
        QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 15px; font-weight: bold; background-color: #2b2b2b; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #2b2b2b; color: #00bcd4; }
        QPushButton { background-color: #3c3f41; border: 1px solid #555; border-radius: 4px; padding: 6px; color: white; font-weight: bold;}
        QPushButton:hover { background-color: #484b4d; }
        QPushButton:disabled { background-color: #222222; color: #666666; }
        QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
        QTabBar::tab { background: #2b2b2b; color: #888888; padding: 8px 25px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: bold; }
        QTabBar::tab:selected { background: #3c3f41; color: #ffffff; border-bottom: 3px solid #00bcd4; }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #3c3f41; border: 1px solid #555; padding: 5px; color: white; border-radius: 3px; }
        QProgressBar { border: 1px solid #555; text-align: center; color: white; border-radius: 3px;}
        QProgressBar::chunk { background-color: #00bcd4; }
        QCheckBox, QRadioButton { color: white; spacing: 5px; }
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
        self.setup_grafico_temporal(); l_time.addWidget(self.can_time)
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
        self.setup_grafico_fft(); l_fft.addWidget(self.can_fft)
        self.tabs.addTab(self.tab_fft, "FFT")

        self.tab_config = QWidget()
        self.setup_aba_configuracoes()
        self.tabs.addTab(self.tab_config, "Configurações e Perfil")

    def setup_aba_configuracoes(self):
        layout = QVBoxLayout(self.tab_config)
        layout_top = QHBoxLayout()

        group_sujeito = QGroupBox("Dados do Participante")
        form_sujeito = QFormLayout()
        self.input_nome = QLineEdit("Participante_01")
        form_sujeito.addRow("Nome do Sujeito:", self.input_nome)
        group_sujeito.setLayout(form_sujeito)
        layout_top.addWidget(group_sujeito)

        self.btn_gravar_paradigma = QPushButton("🎯 Iniciar Protocolo de Paradigma\n(Mestre)")
        self.btn_gravar_paradigma.setStyleSheet("background-color: #ff9800; color: black; font-weight: bold; font-size: 14px; padding: 15px; border-radius: 5px;")
        self.btn_gravar_paradigma.clicked.connect(self.abrir_gravacao_paradigma)
        layout_top.addWidget(self.btn_gravar_paradigma)
        layout.addLayout(layout_top)

        group_gabarito = QGroupBox("Gabarito e Parâmetros da Sessão")
        form_gabarito = QFormLayout()
        str_gabarito = ",".join(map(str, GABARITO_SESSAO))
        self.input_gabarito = QLineEdit(str_gabarito)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 100); self.spin_epochs.setValue(EPOCHS_TREINO)
        self.spin_shape_time = QSpinBox(); self.spin_shape_time.setRange(10, 5000); self.spin_shape_time.setValue(721); self.spin_shape_time.setSuffix(" pts")
        self.spin_shape_ch = QSpinBox(); self.spin_shape_ch.setRange(1, 32); self.spin_shape_ch.setValue(16); self.spin_shape_ch.setSuffix(" ch")
        form_gabarito.addRow("Sequência (Gabarito):", self.input_gabarito)
        form_gabarito.addRow("Épocas de Treino (TL):", self.spin_epochs)
        form_gabarito.addRow("Time Steps (Shape):", self.spin_shape_time)
        form_gabarito.addRow("Canais (Shape):", self.spin_shape_ch)
        group_gabarito.setLayout(form_gabarito); layout.addWidget(group_gabarito)

        group_modelo = QGroupBox("Gestão do Modelo de Inteligência Artificial")
        layout_modelo = QVBoxLayout()
        self.btn_salvar_modelo = QPushButton("💾 Salvar Modelo Atualizado (Fine-Tuned)")
        self.btn_salvar_modelo.setStyleSheet("background-color: #1976d2; font-size: 14px; padding: 8px;")
        self.btn_salvar_modelo.clicked.connect(self.salvar_modelo_novo)
        layout_modelo.addWidget(self.btn_salvar_modelo)
        group_modelo.setLayout(layout_modelo); layout.addWidget(group_modelo)
        layout.addStretch()

    def abrir_gravacao_paradigma(self):
        config_win = JanelaConfiguracaoParadigma(self.conectado_unity)
        if config_win.exec_() == QDialog.Accepted:
            configs = config_win.configs
            paradigma_win = JanelaExecucaoParadigma(configs, unity_sender=self.unity)
            paradigma_win.exec_() 
            novo_gabarito = paradigma_win.sequencia_trials
            if novo_gabarito:
                self.input_gabarito.setText(",".join(map(str, novo_gabarito)))
                QMessageBox.information(self, "Sucesso", "Gravação concluída!\nGabarito gerado inserido.")

    def salvar_modelo_novo(self):
        if not self.model: return QMessageBox.warning(self, "Aviso", "Não há modelo carregado.")
        fname, _ = QFileDialog.getSaveFileName(self, 'Salvar Modelo', f"modelo_TL_{self.input_nome.text()}.h5", "H5 (*.h5)")
        if fname:
            try: self.model.save(fname); QMessageBox.information(self, "Sucesso", "Modelo salvo!")
            except Exception as e: QMessageBox.critical(self, "Erro", str(e))

    def setup_grafico_temporal(self):
        self.ax_time = self.fig_time.add_subplot(111)
        self.fig_time.patch.set_facecolor('#ffffff'); self.ax_time.set_facecolor('#ffffff')
        self.ax_time.tick_params(colors='#333333'); self.ax_time.set_xlim(0, self.x_size); self.ax_time.set_yticks([])
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
        self.ax_fft.set_ylim(0.1, 100); self.ax_fft.set_xlim(0, 60); self.ax_fft.grid(True, which='both', color='#dddddd')
        colors = ['#555555', '#8959a8', '#3e999f', '#71c671', '#e8c346', '#e68136', '#d84e4e', '#8c564b']
        self.lines_fft = [self.ax_fft.plot([],[], lw=1.5, alpha=0.8, color=colors[i%8])[0] for i in range(self.n_channels_hardware)]

    def atualizar_limites_temporal(self):
        self.ax_time.set_ylim(-self.escala_visual, (self.n_channels_hardware * self.escala_visual) + self.escala_visual)

    def carregar_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Abrir CSV OpenBCI', '', "CSV (*.csv)")
        if fname:
            try:
                df = pd.read_csv(fname, comment='%')
                self.dados_arquivo = df.iloc[:, 1 : self.n_channels_hardware + 1].values
                self.ponteiro_arquivo = 0; self.modo_arquivo = True
                self.lbl_csv.setText(f"{fname.split('/')[-1]} ({len(self.dados_arquivo)} pts)"); self.lbl_csv.setStyleSheet("color: #00e676;")
                self.btn_csv.setEnabled(False)
            except Exception as e: QMessageBox.critical(self, "Erro", str(e))

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
        else: self.lbl_lsl.setText("Erro")

    def setup_menu(self):
        self.menuBar().addMenu("Arquivo").addAction("Carregar Modelo IA Base").triggered.connect(self.carregar_modelo_arquivo)

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
        try:
            nova_lista = [int(x.strip()) for x in self.input_gabarito.text().split(',')]
            if len(nova_lista) == 0: raise Exception
            self.gabarito_sessao = nova_lista; self.total_tentativas = len(self.gabarito_sessao)
        except: return QMessageBox.critical(self, "Erro", "Gabarito inválido.")

        self.modo_teste_unity = self.chk_teste_unity.isChecked()
        if not self.modo_teste_unity and not self.inlet and not self.modo_arquivo: return
        
        self.qtd_tl = int(self.total_tentativas * PORCENTAGEM_TL) if self.radio_com_tl.isChecked() else 0
        self.indice_atual = 0; self.acertos_fase1 = 0; self.acertos_fase2 = 0
        self.sessao_iniciada = True; self.btn_iniciar.setEnabled(False)
        self.bar_progresso.setMaximum(self.total_tentativas); self.bar_progresso.setValue(0)
        self.tabs.setCurrentIndex(0)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(10)

    def update_loop(self):
        if self.modo_teste_unity:
            data = np.random.randn(3, self.n_channels_hardware) * 50 
            self.sincronizado = True; self.buffer_sobra.extend(data)
            self.current_data_visual = np.roll(self.current_data_visual, -3, axis=0); self.current_data_visual[-3:, :] = data
        elif self.modo_arquivo:
            chunk_size = 3 
            if self.ponteiro_arquivo + chunk_size < len(self.dados_arquivo):
                data = self.dados_arquivo[self.ponteiro_arquivo : self.ponteiro_arquivo + chunk_size]
                self.ponteiro_arquivo += chunk_size; self.sincronizado = True; self.buffer_sobra.extend(data)
                self.current_data_visual = np.roll(self.current_data_visual, -len(data), axis=0); self.current_data_visual[-len(data):, :] = data
            else: return self.finalizar_sessao()
        else:
            chunk, _ = self.inlet.pull_chunk(timeout=0.0)
            if chunk:
                data = np.array(chunk)[:, :self.n_channels_hardware]
                if not self.sincronizado:
                    if np.sum(np.abs(data)) > 1e-6: self.sincronizado = True
                    else: return
                self.buffer_sobra.extend(data)
                self.current_data_visual = np.roll(self.current_data_visual, -len(data), axis=0); self.current_data_visual[-len(data):, :] = data

        self.atualizar_graficos_visuais()
        
        target_time = self.spin_shape_time.value(); target_ch = self.spin_shape_ch.value()
        while len(self.buffer_sobra) >= target_time:
            if self.modo_teste_unity and self.indice_atual >= self.total_tentativas: self.indice_atual = 0
            elif self.indice_atual >= self.total_tentativas: return self.finalizar_sessao()
            
            raw_epoch = np.array(self.buffer_sobra[:target_time])
            self.buffer_sobra = self.buffer_sobra[target_time:] 
            self.processar_caixa(raw_epoch[:, :target_ch])

    def processar_caixa(self, dados):
        pred = 2; label_real = self.gabarito_sessao[self.indice_atual]; prob_left, prob_right, prob_rest = 0.0, 0.0, 0.0

        if self.modo_teste_unity:
            pred = random.randint(0, 2); self.lbl_fase.setText("MODO TESTE"); self.lbl_fase.setStyleSheet("color: orange;")
            if pred == 0: prob_left = 1.0
            elif pred == 1: prob_right = 1.0
            else: prob_rest = 1.0
        else:
            dados_norm = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8)
            input_data = np.expand_dims(dados_norm, axis=0).astype(np.float32)
            if self.model:
                try:
                    prob = self.model.predict(input_data, verbose=0)[0]
                    if len(prob) == 3: 
                        pred = np.argmax(prob); prob_left, prob_right, prob_rest = prob[0], prob[1], prob[2]
                except: pass
            
            fase = "TREINO COM IA (TL)" if self.indice_atual < self.qtd_tl else "AVALIAÇÃO/TESTE"
            if self.modo_arquivo: fase += " [ARQUIVO]"
            self.lbl_fase.setText(fase); self.lbl_fase.setStyleSheet(f"color: {'yellow' if 'TREINO' in fase else '#00e676'}; font-weight: bold;")

        self.gauge.set_probabilities(prob_left, prob_right, prob_rest)
        acertou = (pred == label_real); self.indice_atual += 1
        
        self.lbl_progresso.setText(f"Tentativa: {self.indice_atual} / {self.total_tentativas}")
        self.bar_progresso.setValue(self.indice_atual)
        nomes = ["ESQUERDA", "DIREITA", "REPOUSO"]; cores = ["#00bcd4", "#ff4081", "#ffffff"]
        self.lbl_predicao.setText(nomes[pred]); self.lbl_predicao.setStyleSheet(f"color: {cores[pred]}")
        self.lbl_feedback.setText("✓ ACERTOU" if acertou else "✗ ERROU"); self.lbl_feedback.setStyleSheet(f"color: {'#00e676' if acertou else '#ff5555'}")

        if self.conectado_unity:
            if acertou:
                if pred == 0: self.unity.send("LEFT")
                elif pred == 1: self.unity.send("RIGHT")
                else: self.unity.send("REST")
            else: self.unity.send("REST")

        if not self.modo_teste_unity and self.indice_atual < self.qtd_tl and self.model:
            if acertou: self.acertos_fase1 += 1
            d_norm = (dados - dados.min()) / (dados.max() - dados.min() + 1e-8)
            inp = np.expand_dims(d_norm, axis=0).astype(np.float32)
            for _ in range(self.spin_epochs.value()): self.model.train_on_batch(inp, np.array([label_real]).astype(np.float32))
        elif not self.modo_teste_unity and acertou: self.acertos_fase2 += 1

    def finalizar_sessao(self):
        self.timer.stop(); self.btn_iniciar.setEnabled(True); self.sessao_iniciada = False
        if not self.modo_teste_unity:
            total_teste = self.total_tentativas - self.qtd_tl
            acc = (self.acertos_fase2 / total_teste)*100 if total_teste > 0 else 0
            msg = QMessageBox()
            msg.setWindowTitle("Fim da Sessão")
            msg.setText(f"Sessão Concluída!\n\nAcurácia na Fase de Teste: {acc:.2f}%")
            msg.exec_()

    def atualizar_graficos_visuais(self):
        if self.tabs.currentIndex() == 0: 
            if self.escala_auto:
                amp = np.ptp(self.current_data_visual, axis=0).max()
                if amp > 1: self.escala_visual = amp * 0.8; self.atualizar_limites_temporal()
            x = np.arange(self.x_size)
            for i, l in enumerate(self.lines_time):
                off = i * self.escala_visual; y = self.current_data_visual[:, i] - np.mean(self.current_data_visual[:, i])
                l.set_data(x, y + off); self.rms_texts[i].set_text(f"{np.sqrt(np.mean(y**2)):.2f} uVrms"); self.rms_texts[i].set_position((self.x_size+10, off))
            self.can_time.draw_idle()
        elif self.tabs.currentIndex() == 1: 
            xf = np.linspace(0, self.fs/2, self.x_size//2)
            for i, l in enumerate(self.lines_fft):
                raw = 2.0/self.x_size * np.abs(fft(self.current_data_visual[:, i])[0:self.x_size//2])
                f = self.fft_smooth_factor; self.fft_buffer_history[i] = (self.fft_buffer_history[i]*f) + (raw*(1-f))
                l.set_data(xf, self.fft_buffer_history[i])
            self.can_fft.draw_idle()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = JanelaInicial()
    win.show()
    sys.exit(app.exec_())