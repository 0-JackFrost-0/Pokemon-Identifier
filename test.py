import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import cv2
import torchvision
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor, Lambda
from models import NeuralNetwork1, NeuralNetwork2
from datasets import TrainingDataset
import numpy as np
import argparse



device = "cpu"


# Hyperparameters
load = True

model = NeuralNetwork1().to(device)

if load:
    model.load_state_dict(torch.load("output/output.pth"))
    model.eval()

class_map = {"Abra" : 0 ,
                            "Aerodactyl" : 1 ,
                            "Alakazam" : 2 ,
                            "Alolan Sandslash" : 3 ,
                            "Arbok" : 4 ,
                            "Arcanine" : 5 ,
                            "Articuno" : 6 ,
                            "Beedrill" : 7 ,
                            "Bellsprout" : 8 ,
                            "Blastoise" : 9 ,
                            "Bulbasaur" : 10 ,
                            "Butterfree" : 11 ,
                            "Caterpie" : 12 ,
                            "Chansey" : 13 ,
                            "Charizard" : 14 ,
                            "Charmander" : 15 ,
                            "Charmeleon" : 16 ,
                            "Clefable" : 17 ,
                            "Clefairy" : 18 ,
                            "Cloyster" : 19 ,
                            "Cubone" : 20 ,
                            "Dewgong" : 21 ,
                            "Diglett" : 22 ,
                            "Ditto" : 23 ,
                            "Dodrio" : 24 ,
                            "Doduo" : 25 ,
                            "Dragonair" : 26 ,
                            "Dragonite" : 27 ,
                            "Dratini" : 28 ,
                            "Drowzee" : 29 ,
                            "Dugtrio" : 30 ,
                            "Eevee" : 31 ,
                            "Ekans" : 32 ,
                            "Electabuzz" : 33 ,
                            "Electrode" : 34 ,
                            "Exeggcute" : 35 ,
                            "Exeggutor" : 36 ,
                            "Farfetchd" : 37 ,
                            "Fearow" : 38 ,
                            "Flareon" : 39 ,
                            "Gastly" : 40 ,
                            "Gengar" : 41 ,
                            "Geodude" : 42 ,
                            "Gloom" : 43 ,
                            "Golbat" : 44 ,
                            "Goldeen" : 45 ,
                            "Golduck" : 46 ,
                            "Golem" : 47 ,
                            "Graveler" : 48 ,
                            "Grimer" : 49 ,
                            "Growlithe" : 50 ,
                            "Gyarados" : 51 ,
                            "Haunter" : 52 ,
                            "Hitmonchan" : 53 ,
                            "Hitmonlee" : 54 ,
                            "Horsea" : 55 ,
                            "Hypno" : 56 ,
                            "Ivysaur" : 57 ,
                            "Jigglypuff" : 58 ,
                            "Jolteon" : 59 ,
                            "Jynx" : 60 ,
                            "Kabuto" : 61 ,
                            "Kabutops" : 62 ,
                            "Kadabra" : 63 ,
                            "Kakuna" : 64 ,
                            "Kangaskhan" : 65 ,
                            "Kingler" : 66 ,
                            "Koffing" : 67 ,
                            "Krabby" : 68 ,
                            "Lapras" : 69 ,
                            "Lickitung" : 70 ,
                            "Machamp" : 71 ,
                            "Machoke" : 72 ,
                            "Machop" : 73 ,
                            "Magikarp" : 74 ,
                            "Magmar" : 75 ,
                            "Magnemite" : 76 ,
                            "Magneton" : 77 ,
                            "Mankey" : 78 ,
                            "Marowak" : 79 ,
                            "Meowth" : 80 ,
                            "Metapod" : 81 ,
                            "Mew" : 82 ,
                            "Mewtwo" : 83 ,
                            "Moltres" : 84 ,
                            "MrMime" : 85 ,
                            "Muk" : 86 ,
                            "Nidoking" : 87 ,
                            "Nidoqueen" : 88 ,
                            "Nidorina" : 89 ,
                            "Nidorino" : 90 ,
                            "Ninetales" : 91 ,
                            "Oddish" : 92 ,
                            "Omanyte" : 93 ,
                            "Omastar" : 94 ,
                            "Onix" : 95 ,
                            "Paras" : 96 ,
                            "Parasect" : 97 ,
                            "Persian" : 98 ,
                            "Pidgeot" : 99 ,
                            "Pidgeotto" : 100 ,
                            "Pidgey" : 101 ,
                            "Pikachu" : 102 ,
                            "Pinsir" : 103 ,
                            "Poliwag" : 104 ,
                            "Poliwhirl" : 105 ,
                            "Poliwrath" : 106 ,
                            "Ponyta" : 107 ,
                            "Porygon" : 108 ,
                            "Primeape" : 109 ,
                            "Psyduck" : 110 ,
                            "Raichu" : 111 ,
                            "Rapidash" : 112 ,
                            "Raticate" : 113 ,
                            "Rattata" : 114 ,
                            "Rhydon" : 115 ,
                            "Rhyhorn" : 116 ,
                            "Sandshrew" : 117 ,
                            "Sandslash" : 118 ,
                            "Scyther" : 119 ,
                            "Seadra" : 120 ,
                            "Seaking" : 121 ,
                            "Seel" : 122 ,
                            "Shellder" : 123 ,
                            "Slowbro" : 124 ,
                            "Slowpoke" : 125 ,
                            "Snorlax" : 126 ,
                            "Spearow" : 127 ,
                            "Squirtle" : 128 ,
                            "Starmie" : 129 ,
                            "Staryu" : 130 ,
                            "Tangela" : 131 ,
                            "Tauros" : 132 ,
                            "Tentacool" : 133 ,
                            "Tentacruel" : 134 ,
                            "Vaporeon" : 135 ,
                            "Venomoth" : 136 ,
                            "Venonat" : 137 ,
                            "Venusaur" : 138 ,
                            "Victreebel" : 139 ,
                            "Vileplume" : 140 ,
                            "Voltorb" : 141 ,
                            "Vulpix" : 142 ,
                            "Wartortle" : 143 ,
                            "Weedle" : 144 ,
                            "Weepinbell" : 145 ,
                            "Weezing" : 146 ,
                            "Wigglytuff" : 147 ,
                            "Zapdos" : 148 ,
                            "Zubat" : 149
                        }
pok_map = {v: k for k, v in class_map.items()}

path = "./test/img.jpg"
img = cv2.imread(path, 0)
img = cv2.resize(img, (416, 416))
img_tensor = torchvision.transforms.functional.to_tensor(img)
img_tensor = img_tensor.permute(2, 0, 1)
pred = model(img_tensor)
print(pok_map[pred])
