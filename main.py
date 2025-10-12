from config import *
from modelo import carregar_modelo
from conec_unity import configurar_socket
from acquisition import normalize_sample, Sistema
from plot import atualizar_grafico
import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
from pylsl import StreamInlet, resolve_stream
from time import sleep

if __name__ == '__main__':
    # Configura rede e ZMQ
    socket_pub, context = configurar_socket()

    # Carrega modelo
    model = carregar_modelo()
    print(f"Aguardando stream EEG... input shape: {model.input_shape}")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    sleep(1)
    print("Stream EEG encontrada!")

    # Inicializações
    sistema = Sistema()
    epochsize = model.input_shape[1]
    tgraf, buffer_x, buffer_y = [], [], []
    task2label = {'T1': 0, 'T2': 1}
    current_data = []
    started = False
    fase_atual = 1
    tempo_inicio = None

    if fase1_forced_label not in ['T1','T2']:
        raise ValueError("fase1_forced_label deve ser 'T1' ou 'T2'.")
    print(f"Fase 1 configurada para aceitar sempre: {fase1_forced_label}")

    plt.ion()
    fig, ax = plt.subplots()
    last_plot = time.time()

    print('Pressione ESC para encerrar.')
    while not keyboard.is_pressed('Esc'):
        chunk, _ = inlet.pull_chunk()
        if not chunk:
            continue
        for sample in chunk:
            # Início detecção
            if not started and np.any(np.array(sample) != 0):
                started = True
                tempo_inicio = time.time()
                print("Início da Fase 1: dados válidos detectados.")
            if not started:
                continue

            # Controle de fases
            elapsed = time.time() - tempo_inicio
            if fase_atual == 1 and elapsed > tempo_fase1:
                fase_atual, tempo_inicio = 2, time.time(); print("-> Fase 2 iniciada")
            elif fase_atual == 2 and elapsed > tempo_fase2:
                fase_atual, tempo_inicio = 3, time.time(); print("-> Fase 3 iniciada")
            elif fase_atual == 3 and elapsed > tempo_fase3:
                print("-> Todas as fases concluídas. Encerrando loop."); break

            # Acumula dados
            current_data.append(sample)
            if len(current_data) == epochsize:
                arr_norm = normalize_sample(current_data)
                pred = model(np.expand_dims(arr_norm, axis=0), training=False).numpy()[0][0]

                # Define label
                if pred < limiar:
                    label = 'T1'
                elif pred > 1 - limiar:
                    label = 'T2'
                else:
                    label = 'T0'

                # Envia comandos garantindo estado exclusivo
                if label == 'T1':
                    socket_pub.send_string("LEFT_HAND_CLOSE")
                    socket_pub.send_string("RIGHT_HAND_OPEN")
                elif label == 'T2':
                    socket_pub.send_string("RIGHT_HAND_CLOSE")
                    socket_pub.send_string("LEFT_HAND_OPEN")
                else:  # T0
                    socket_pub.send_string("LEFT_HAND_OPEN")
                    socket_pub.send_string("RIGHT_HAND_OPEN")

                # Transfer Learning
                do_tl = False
                if fase_atual == 1 and label == fase1_forced_label:
                    do_tl = True
                elif fase_atual == 2 and label != fase1_forced_label and label in ('T1','T2'):
                    do_tl = True
                if do_tl and label in ('T1','T2'):
                    buffer_x.append(arr_norm); buffer_y.append(task2label[label])
                    if len(buffer_x) >= batch_size:
                        print("=== Iniciando transfer learning (1 época) ===")
                        model.fit(np.array(buffer_x), np.array(buffer_y).reshape(-1,1), epochs=1, verbose=1)
                        print("=== Transfer learning concluído ===")
                        buffer_x, buffer_y = [], []

                tgraf.append(label)
                print(f"[Fase {fase_atual}] Previsão: {label} | Prob: {pred:.3f} | TL: {'Sim' if do_tl else 'Não'}")
                current_data = []

            # Atualiza gráfico
            if time.time() - last_plot >= sistema.dt:
                last_plot = time.time()
                atualizar_grafico(ax, tgraf, fase_atual)
        else:
            continue
        break

    plt.ioff(); plt.show()
    socket_pub.close(); context.term()
