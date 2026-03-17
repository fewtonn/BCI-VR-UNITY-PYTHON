BCI-VR-UNITY-PYTHON 🧠🕶️
Simulation of a Brain-Computer Interface (BCI) integrated with a Virtual Reality environment using Unity and Python communication.

🎮 Unity VR Space
To access the 3D UNITY space and connect it to this controller, download the project file here:
👉 [Download Unity BCI-VR Project](https://drive.google.com/file/d/1qv33TthXWdQOnMN3auH8BM6PIl_oNDcc/view?usp=sharing)

🛠️ Data Pre-processing: Do .mat para OpenBCI CSV
Além do controle em tempo real, este repositório contém scripts para converter datasets de pesquisa (formato MATLAB .mat) para o formato padrão do OpenBCI GUI. Isso permite que você treine seus modelos com dados históricos como se estivessem vindo diretamente do capacete.

Passo a Passo do Código de Conversão:
O script realiza as seguintes etapas para garantir a compatibilidade dos dados:

1. Configurações de Canais e Mapeamento
O código define a origem dos dados (21 canais padrão de sistemas clínicos) e filtra apenas os 16 canais compatíveis com a placa OpenBCI Cyton + Daisy.

Mapeamento de Classes: Converte os marcadores do dataset (ex: 1, 2, 3) para labels de processamento (0: Repouso, 1: Esquerda, 2: Direita).

2. Localização Automática de Sinais (encontrar_indices_automaticamente)
Como arquivos .mat podem ter estruturas diferentes, esta função varre o arquivo procurando por:

Matrizes que possuam entre 21 e 22 linhas (indicando canais de EEG).

Vetores com valores baixos (indicando os marcadores/eventos de cada tentativa).

Valores escalares que representem a frequência de amostragem (Ex: 250Hz ou 200Hz).

3. Extração de Épocas (extrair_epocas_mat)
O script percorre todos os arquivos de uma pasta, identifica onde um movimento começou (mudança no canal de marcadores) e recorta um "pedaço" do sinal (época) de tempo pré-definido (neste caso, 1 segundo após o estímulo).

4. Formatação Estilo OpenBCI (salvar_csv_openbci)
Esta é a parte crucial para a compatibilidade. O script não apenas salva os números, mas reconstrói a estrutura de um arquivo gerado pelo software oficial do OpenBCI:

Cabeçalho (Header): Adiciona as linhas de comentário %OpenBCI Raw EXG Data, %Sample Rate, etc.

Colunas Extras: Cria colunas falsas de Acelerômetro, Canais Analógicos e Timestamps formatados, para que o leitor de CSV do seu modelo de IA não perceba a diferença entre um dado real e um dado convertido.

5. Exportação de Labels
Gera um arquivo labels_ordem.txt que serve como o "gabarito" para o treino da sua rede neural, indicando exatamente o que o usuário estava pensando em cada segundo do arquivo CSV gerado.

🚀 Como usar a conversão
Coloque seus arquivos .mat na pasta configurada em DIRETORIO_RAIZ.

Execute o script de conversão.

Use o arquivo ARQUVOOPENBCI_16CANAIS.csv resultante no seu pipeline de treinamento ou simulação de sinal.

Dica para o seu projeto:
Ao subir o código para o GitHub, você pode organizar assim:

main.py: O código da interface BCI (que você me enviou antes).

converter_mat_to_csv.py: Este novo código de processamento.

requirements.txt: Com as bibliotecas que adicionamos hoje (scipy e pandas).

Deseja que eu ajude a criar uma pequena "Wiki" dentro do README explicando como configurar o IP do Unity para que ele receba os comandos do Python sem erro?