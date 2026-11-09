from neuro_SGD import Neuroverkko
import matplotlib.pyplot as plt
import tensorflow as tf

#hyperparametrit
kerrokset_neuronit=[784, 256, 128, 10]
aktivointifunktio=["Relu","relu_leaky"][0]
momentum_optimoiniti=True
l2_menetelma=False
epokki=30
erakoko=128
oppimis_nopeus=0.01
tulostiedosto="tulostiedosto.pkl"
painotiedosto="painotiedosto.pkl"

#instanssi Neuroverkosta
verkko = Neuroverkko(
    kerrokset=kerrokset_neuronit,
    aktivointi=aktivointifunktio, 
    momentum = momentum_optimoiniti, 
    L2 = l2_menetelma
)

#koulutus
verkko.kouluta(
    epochs=epokki,
    era_koko=erakoko,
    oppimisnopeus=oppimis_nopeus
)

#tallenna painot ja vakiokertoimet
verkko.tallenna_malli(painotiedosto)
#tallenna tulokset
verkko.tallenna_tulokset(tulostiedosto)
#haetaan tulokset
verkko_tulokset=verkko.lataa_tulokset(tulostiedosto)

#piirretaan kuvia tuloksista
#luokittelutarkkuus testiaineistolla
plt.figure(figsize=(10, 6))
plt.plot(verkko_tulokset["epoch"], verkko_tulokset["testi_tarkkuus"],
         color="black", linestyle="-", linewidth=1.4,
         label=r"SGD + Momentum + $L_2$")
plt.title("Testaus tarkkuus")
plt.xlabel("Epoch")
plt.ylabel("Testaus tarkkuus")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#luokittelutarkkuus koulutusaineistolla
plt.figure(figsize=(10, 6))
plt.plot(verkko_tulokset["epoch"], verkko_tulokset["koulutus_tarkkuus"],
         color="black", linestyle="-", linewidth=1.4,
         label=r"SGD + Momentum + $L_2$")
plt.title("Koulutus tarkkuus")
plt.xlabel("Epoch")
plt.ylabel("Koulutus tarkkuus")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


#havio
plt.figure(figsize=(10, 6))
plt.plot(verkko_tulokset["epoch"], verkko_tulokset["havio"],
         color="black", linestyle="-", linewidth=1.4,
         label=r"SGD + Momentum + $L_2$")
plt.title("Havio")
plt.xlabel("Epoch")
plt.ylabel("Havio")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#painot
plt.figure(figsize=(10, 6))
plt.plot(verkko_tulokset["epoch"], verkko_tulokset["painot"],
         color="black", linestyle="-", linewidth=1.4,
         label=r"SGD + Momentum + $L_2$")
plt.title("Painot")
plt.xlabel("Epoch")
plt.ylabel("Painot")
plt.legend(frameon=False)
plt.grid(alpha=0.3)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#uusi instanssi Neuroverkosta
verkko_uusi = Neuroverkko(
    kerrokset=kerrokset_neuronit,
    aktivointi=aktivointifunktio, 
    momentum = momentum_optimoiniti, 
    L2 = l2_menetelma)

#ladataan painot, koulutusta ei tarvitse tehda
verkko_uusi.lataa_malli(painotiedosto)

#ladataan MNIST-aineisto, normalisoidaan se ja muutetaan vektoriksi
(x_train, y_train), (x_test1, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test1 = x_test1 / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test1.reshape(-1, 784)

#kuvien valinta
kuvat=[1000, 1, 3, 98, 2000, 2001, 2002]
kuvat=range(1,500)

#tehdaan luokittelu, mikali luokittelutulos on vaara, numero piirretaan
for i in kuvat:
    #luokittelu
    nr1=verkko.luokittele(x_test[i])
    nr2=verkko_uusi.luokittele(x_test[i])
    #tulokset
    print(
        f"Luokittelun nr1 tulos {nr1}, "
        f"Luokittelun nr2 tulos {nr2}, "
        f"oikea numero {y_test[i]}"
    )
    if nr1 != y_test[i]:
        mustavalkoinen = (x_test1[i] > 0.2)
        plt.imshow(mustavalkoinen, cmap='binary')
        plt.title(f"Luokittelun tulos {nr1}, oikea numero {y_test[i]}")
        plt.axis('off')
        plt.show()