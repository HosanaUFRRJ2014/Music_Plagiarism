#!/usr/bin/python
# -*- coding: utf-8 -*-

get_ipython().magic(u'matplotlib inline')
import seaborn
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas
import IPython.display as ipd
import librosa
import librosa.display
import random
import numpy as np
from numpy import dot as matrixmultiply
from scipy.signal import hilbert, chirp
import matplotlib as mpl
plt.rcParams['figure.figsize'] = (16, 8)

# carregar amostra original

(x, sr) = librosa.load('Soul Makossa - Manu Dibango.wav')  # x = matriz de áudio e sr=taxa de amostragem
ipd.Audio(x, rate=sr)

# primeiro, fixar parâmetros STFT

hop_length = 512
n_fft = 2048

# Calcuar STFT

S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
X = abs(S) ** 2  # obter magnitude do espectograma

# definir parâmetros NMF

numComp = 3  # número de componentes
numIter = 30  # n de interações
(numBins, numFrames) = X.shape

# primeiro, o treinamento NMF no espectrograma da amostra original
# inicializar

B = np.random.rand(numBins, numComp)  # matriz
G = np.random.rand(numComp, numFrames)  # matriz
OnesMat = numpy.ones((numBins, numFrames), dtype=np.float)  # cria uma matriz numBis X numFrames de 1's

# Plotar componentes NFM

duration = 1.0
fs = 244.0
samples = int(fs * duration)
t = np.arange(samples) / fs

analytic_signal = hilbert(Glinha)
amplitude_envelope = np.abs(analytic_signal)

fig = plt.figure(figsize=(15, 10))
ax0 = fig.add_subplot(211)
plt.gca().set_color_cycle(['red', 'blue', 'green'])
ax0.plot(t, amplitude_envelope)
plt.show()

for it in range(1, numIter):
    print it
    Xtilde = np.spacing(1) + matrixmultiply(B, G)  # atualizando o modelo
    Xratio = X / Xtilde  # atualização de taxa

    # ativações de atualização

    Blinha = B.conj().transpose()
    G = G * matrixmultiply(Blinha, Xratio) / matrixmultiply(Blinha,
            OnesMat)
    Xtilde = np.spacing(1) + matrixmultiply(B, G)
    Xratio = X / Xtilde
    Glinha = G.conj().transpose()
    B = B * matrixmultiply(Xratio, Glinha) / matrixmultiply(OnesMat,Glinha)
    normB = np.spacing(1) + B.sum(axis=0)
    B = np.multiply(B, 1 / normB)

    # Plotar as componentes NFM

    duration = 1.0
    fs = 244.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    analytic_signal = hilbert(Glinha)
    amplitude_envelope = np.abs(analytic_signal)
    fig = plt.figure(figsize=(15, 10))
    ax0 = fig.add_subplot(211)
    plt.gca().set_color_cycle(['red', 'blue', 'green'])
    ax0.plot(t, amplitude_envelope)
    plt.savefig('ultimaIterOriginal.png')
    plt.show()

# Plotar componentes separadas amostra original

plt.figure()
for n in range(numComp):
    plt.subplot(numpy.ceil(numComp), 2, n + 1)
    plt.gca().set_color_cycle(['blue'])
    plt.plot(G[n])
    plt.ylim(0, G.max())
    plt.xlim(0, G.shape[1])
    plt.ylabel('Componente %d' % n)
    plt.savefig('dibangocomp.png')
G1 = G

# Reproduzir o sinal de cada componente

reconstructed_signal = scipy.zeros(len(x))
for n in range(numComp):
    Y = scipy.outer(B[:, n], G[n]) * numpy.exp(1j * numpy.angle(S))
    y = librosa.istft(Y)
    reconstructed_signal[:len(y)] += y
    ipd.display(ipd.Audio(y, rate=sr))

# Em seguida, repare os modelos aprendidos e aplique-os ao suspeito de plágio mais alguns componentes extras

storeB = B  # componentes extras
numCompFixed = numComp
numComp = 3

# carregar amostra suspeita

(x, sr) = librosa.load('Michael Jackson -Wanna Be Startin Somethin.wav')  # x= matriz de áudio e sr=taxa de amostragem
ipd.Audio(x, rate=sr)

# Plotar componentes NFM

duration = 1.0
fs = 244.0
samples = int(fs * duration)
t = np.arange(samples) / fs

analytic_signal = hilbert(Glinha)
amplitude_envelope = np.abs(analytic_signal)

fig = plt.figure(figsize=(15, 10))
ax0 = fig.add_subplot(211)
plt.gca().set_color_cycle(['red', 'blue', 'green'])
ax0.plot(t, amplitude_envelope)
plt.show()

# Calcuar STFT

S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
X = abs(S) ** 2  # obter magnitude do espectograma
(numBins, numFrames) = X.shape

# inicializar

B = np.random.rand(numBins, numComp)
G = np.random.rand(numComp, numFrames)
OnesMat = numpy.ones((numBins, numFrames), dtype=np.float)

# inicializar os primeiros modelos com os que já foram aprendidos

B[:, 0:numCompFixed] = storeB

for it in range(1, numIter):
    print it
    Xtilde = np.spacing(1) + matrixmultiply(B, G)
    Xratio = X / Xtilde  # atualização de taxa
    Blinha = B.conj().transpose()  # ativações de atualização
    G = G * matrixmultiply(Blinha, Xratio) / matrixmultiply(Blinha,OnesMat)
    Xtilde = np.spacing(1) + matrixmultiply(B, G)
    Xratio = X / Xtilde
    Glinha = G.conj().transpose()
    B = B * matrixmultiply(Xratio, Glinha) / matrixmultiply(OnesMat,Glinha)
    B[:, 0:numCompFixed] = storeB
    normB = np.spacing(1) + B.sum(axis=0)
    B = np.multiply(B, 1 / normB)

    # Plotar as componentes NFM

    duration = 1.0
    fs = 293.0
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    analytic_signal = hilbert(Glinha)
    amplitude_envelope = np.abs(analytic_signal)
    fig = plt.figure(figsize=(15, 10))
    ax0 = fig.add_subplot(211)
    plt.gca().set_color_cycle(['red', 'blue', 'green'])
    ax0.plot(t, amplitude_envelope)
    plt.ylim(0, 80000)
    plt.savefig('ultimaIterPlagio.png')
    plt.show()

# Plotar as componentes NFM

duration = 1.0
fs = 293.0
samples = int(fs * duration)
t = np.arange(samples) / fs

analytic_signal = hilbert(Glinha)
amplitude_envelope = np.abs(analytic_signal)

fig = plt.figure(figsize=(15, 10))
ax0 = fig.add_subplot(211)
plt.gca().set_color_cycle(['red', 'blue', 'green'])
ax0.plot(t, amplitude_envelope)
plt.show()

# Plotar componentes separadas amostra suspeita

plt.figure()
for n in range(numComp):
    plt.subplot(numpy.ceil(numComp), 2, n + 1)
    plt.gca().set_color_cycle(['red'])
    plt.plot(G[n])
    plt.ylim(0, G.max())
    plt.xlim(0, G.shape[1])
    plt.ylabel('Componente %d' % n)
    plt.savefig('michaelcomp.png')

reconstructed_signal = scipy.zeros(len(x))
for n in range(numComp):
    Y = scipy.outer(B[:, n], G[n]) * numpy.exp(1j * numpy.angle(S))
    y = librosa.istft(Y)
    reconstructed_signal[:len(y)] += y
    ipd.display(ipd.Audio(y, rate=sr))

# Plotar componentes juntas da amostra original e suspeita

plt.figure()
print G.shape
for n in range(numComp):
    plt.subplot(numpy.ceil(numComp), 2, n + 1)
    plt.gca().set_color_cycle(['blue', 'red', 'green'])
    plt.plot(G1[n], label='Amostra original')
    plt.plot(G[n], label='Amostra suspeita ')
    plt.ylim(0, max(G1[n].max(), G[n].max()))
    plt.xlim(0, G.shape[1])
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.ylabel('Componente %d' % n)
    plt.savefig('dibangoemichaelcomp.png')

# Plotar componentes juntas da amostra original e suspeita

plt.figure()
print G.shape
for n in range(numComp):
    plt.subplot(numpy.ceil(numComp), 2, n + 1)
    plt.gca().set_color_cycle(['blue', 'red', 'green'])
    plt.plot(G1[n], label='Amostra original')
    plt.plot(G[n] / 2, label='Amostra suspeita ')
    plt.ylim(0, max(G1[n].max(), (G[n] / 2).max()))
    plt.xlim(0, G.shape[1])
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.ylabel('Componente %d' % n)
    plt.savefig('dibangoemichaelcomp2.png')
