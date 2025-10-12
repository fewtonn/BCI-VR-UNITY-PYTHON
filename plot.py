import matplotlib.pyplot as plt
from config import *

def atualizar_grafico(ax, tgraf, fase):
    ax.cla(); ax.plot(tgraf[-100:], marker='o', linestyle='-')
    ax.set_title(f"Live: Fase {fase} | Label Fase1: {fase1_forced_label}")
    plt.draw(); plt.pause(0.001)