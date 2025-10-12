import numpy as np
import keyboard
import time
import threading
from keras.models import load_model
from pylsl import StreamInlet, resolve_stream
from time import sleep
import zmq
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import os

# === CONFIGURAÇÕES ===
MODEL_PATH = "C:\\Users\\batis\\Downloads\\melhor_modelo_0.8871.h5"
FASE1_FORCED_LABEL = 'T1'
TEMPO_FASE1 = 28.0
TEMPO_FASE2 = 24.5
TEMPO_FASE3 = 124.99
LIMIAR = 0.4
BATCH_SIZE = 1
ARQ_EPOCAS = "todas_epocas.txt"

# === ZMQ SETUP ===
context = zmq.Context()
socket_pub = context.socket(zmq.PUB)
socket_pub.bind("tcp://*:5555")

# === Funções auxiliares ===
def normalize_sample(input_sample):
    inp = np.array(input_sample)
    local_max, local_min = inp.max(), inp.min()
    return (inp - local_min) / (local_max - local_min + 1e-8)

# === Plot em thread separada ===
class LivePlot:
    def __init__(self):
        self.labels = []
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self.plot_loop, daemon=True).start()

    def update(self, label):
        with self.lock:
            self.labels.append(label)

    def plot_loop(self):
        plt.ion()
        fig, ax = plt.subplots()
        while self.running:
            with self.lock:
                if len(self.labels) == 0:
                    continue
                ax.clear()
                ax.plot(self.labels[-100:], marker='o')
                ax.set_title("Últimos 100 rótulos")
                plt.pause(0.01)
        plt.ioff()
        plt.close()

    def stop(self):
        self.running = False

# === ENVIO UNITY EM THREAD ===
class UnitySender:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.sender_loop, daemon=True).start()

    def send(self, msg):
        with self.lock:
            self.queue.append(msg)

    def sender_loop(self):
        while self.running:
            with self.lock:
                if self.queue:
                    msg = self.queue.pop(0)
                    socket_pub.send_string(msg)
            time.sleep(0.01)

    def stop(self):
        self.running = False

# === MAIN ===
def main():
    print("Aguardando stream EEG...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    sleep(1)
    print("Stream EEG conectada.")

    model = load_model(MODEL_PATH)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    epochsize = model.input_shape[1]
    print(f"Esperado tamanho de janela: {epochsize}")

    # Variáveis principais
    fase_atual = 1
    label_fase1 = FASE1_FORCED_LABEL
    tempo_inicio = None
    current_data = []
    started = False
    tgraf = []
    buffer_x, buffer_y = [], []
    task2label = {'T1': 0, 'T2': 1}
    epoch_idx = 0

    # Threads
    plot = LivePlot()
    unity = UnitySender()

    # Arquivo de texto
    with open(ARQ_EPOCAS, 'w', encoding='utf-8') as epoca_txt:
        epoca_txt.write("TODAS AS ÉPOCAS DO EEG (CRUAS)\n")
        epoca_txt.write("="*50 + "\n\n")

        print("Pressione ESC para encerrar.")
        while not keyboard.is_pressed('esc'):
            chunk, _ = inlet.pull_chunk()
            if not chunk:
                continue

            for sample in chunk:
                if not started and np.any(np.array(sample) != 0):
                    started = True
                    tempo_inicio = time.time()
                    print("Início da Fase 1.")

                if not started:
                    continue

                elapsed = time.time() - tempo_inicio
                if fase_atual == 1 and elapsed > TEMPO_FASE1:
                    fase_atual = 2
                    tempo_inicio = time.time()
                    print("-> Fase 2 iniciada.")
                elif fase_atual == 2 and elapsed > TEMPO_FASE2:
                    fase_atual = 3
                    tempo_inicio = time.time()
                    print("-> Fase 3 iniciada.")
                elif fase_atual == 3 and elapsed > TEMPO_FASE3:
                    print("-> Fim das fases.")
                    break

                current_data.append(sample)
                if len(current_data) == epochsize:
                    # ==== SALVAR DADOS CRUS ====
                    epoca_txt.write(f"Época {epoch_idx} - {time.time():.3f}\n")
                    for linha in current_data:
                        epoca_txt.write(','.join(f"{x:.6f}" for x in linha) + "\n")
                    epoca_txt.write("-"*40 + "\n\n")
                    epoca_txt.flush()
                    epoch_idx += 1
                    # ============================

                    arr_norm = normalize_sample(current_data)
                    pred = model(np.expand_dims(arr_norm, axis=0), training=False).numpy()[0][0]
                    if pred < LIMIAR:
                        label = 'T1'
                    elif pred > 1 - LIMIAR:
                        label = 'T2'
                    else:
                        label = 'T0'

                    fez_tl = False
                    if fase_atual == 1 and label == label_fase1:
                        fez_tl = True
                    elif fase_atual == 2 and label != label_fase1 and label in ['T1', 'T2']:
                        fez_tl = True

                    if fez_tl and label in ('T1', 'T2'):
                        buffer_x.append(arr_norm)
                        buffer_y.append(task2label[label])
                        if len(buffer_x) >= BATCH_SIZE:
                            print("Treinando TL...")
                            model.fit(np.array(buffer_x), np.array(buffer_y).reshape(-1,1), epochs=1, verbose=0)
                            buffer_x, buffer_y = [], []

                    # Envia comandos ao Unity via thread
                    if label == 'T1':
                        unity.send("LEFT_HAND_CLOSE")
                        unity.send("RIGHT_HAND_OPEN")
                    elif label == 'T2':
                        unity.send("LEFT_HAND_OPEN")
                        unity.send("RIGHT_HAND_CLOSE")
                    else:
                        unity.send("LEFT_HAND_OPEN")
                        unity.send("RIGHT_HAND_OPEN")

                    # Atualiza o gráfico
                    plot.update(label)
                    print(f"[Fase {fase_atual}] Label: {label} | Prob: {pred:.3f} | TL: {'Sim' if fez_tl else 'Não'}")

                    current_data = []

            else:
                continue
            break  # fim do while principal

    # Finalização
    plot.stop()
    unity.stop()
    socket_pub.close()
    context.term()
    print("Aplicação encerrada.")

if __name__ == "__main__":
    main()
