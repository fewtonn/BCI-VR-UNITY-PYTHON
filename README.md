# BCI-VR-UNITY-PYTHON 🧠🎮

Simulation of a Brain-Computer Interface (BCI) integrated with a Virtual Reality environment using Unity and Python communication.

### 🎮 Unity VR Space
To access the 3D UNITY space and connect it to this controller, download the project file here:
👉 [Download Unity BCI-VR Project](https://drive.google.com/file/d/1qv33TthXWdQOnMN3auH8BM6PIl_oNDcc/view?usp=sharing)

---

## 🚀 Funcionalidades Principais

### 1. BCI Control Center (Python)
A interface principal desenvolvida em **PyQt5** que gerencia todo o fluxo:
* **Conexão LSL (Lab Streaming Layer):** Recebe dados de hardware de EEG (OpenBCI, Cyton, etc).
* **Visualização em Tempo Real:** Gráficos de Série Temporal (uVrms) e FFT (Frequência).
* **IA e Aprendizado Online:** Carregamento de modelos `.h5` (Keras/TensorFlow) com suporte a **Transfer Learning** durante a sessão.
* **Modo Simulação:** Gera sinais aleatórios para testar o Unity sem necessidade de hardware.

### 2. Comunicação de Baixa Latência
O sistema utiliza uma arquitetura híbrida para conversar com o Unity:
* **ZeroMQ (ZMQ):** Protocolo PUB/SUB para envio de comandos diretos (`LEFT`, `RIGHT`, `REST`).
* **UDP Broadcast:** O Python espalha seu IP na rede local para que o Unity o encontre automaticamente, eliminando a necessidade de configurar IPs manualmente.

### 3. Conversor de Dados (Dataset .mat → OpenBCI)
Um script dedicado para preparar dados de pesquisa para a sua rede neural:
* **Filtragem de Canais:** Converte datasets clínicos (21 canais) para o padrão OpenBCI (16 canais).
* **Extração de Épocas:** Recorta automaticamente os sinais baseando-se nos marcadores de evento do MATLAB.
* **Formatação OpenBCI:** Gera um CSV idêntico ao exportado pelo OpenBCI GUI, facilitando o reuso de pipelines de treino.

---

## 🛠️ Instalação e Configuração

### Pré-requisitos
Certifique-se de ter o Python 3.9+ instalado.

```bash
# Clone o repositório
git clone [https://github.com/SEU_USUARIO/BCI-VR-UNITY-PYTHON.git](https://github.com/SEU_USUARIO/BCI-VR-UNITY-PYTHON.git)

# Instale as dependências
pip install pyqt5 numpy scipy matplotlib pylsl pyzmq pandas tensorflow keras
