import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


class TrainingDataset(Dataset):
    def __init__(self):
        self.imgs_path = "PokemonData/"
        directory = os.getcwd() + "/" + self.imgs_path
        self.data = []
        for file in os.listdir(directory):
            curr_path = directory + file + "/"
            for file2 in os.listdir(curr_path):
                self.data.append([directory + file + "/" + file2, file])

        # for class_path in file_list:
        #     class_name = class_path.split("/")[-1]
        #     for img_path in glob.glob(class_path + "/*.jpeg"):
        #         self.data.append([img_path, class_name])
        # print(self.data)

        # self.class_map = {"Abra" : 0, "Aerodactyl": 1,"Alakazam": 2,"Alolan Sandslash":3,"Arbok":4,"Arcanine":5,"Articuno":6,"Beedrill":7,"Bellsprout":8,"Blastoise":9,"Bulbasaur":10,"Butterfree":11,"Caterpie":12,"Chansey":13,"Charizard":14,"Charmander":15,"Charmeleon":16,"Clefable":17,"Clefairy":18,"Cloyster":19,"Cubone":20,"Dewgong":21,"Diglett":22,"Ditto":23,"Dodrio":24,"Doduo":25,"Dragonair": 26,"Dragonite":27,"Dratini":28,"Drowzee":29,"Dugtrio":30,"Eevee":31,"Ekans":32,"Electabuzz":33,"Electrode":34,"Exeggcute":35,"Exeggutor":36,"Farfetchd":37,"Fearow":38,"Flareon":39,"Gastly":40,"Gengar":41,"Geodude":42,"Gloom":43,"Golbat":44,"Goldeen":45,"Golduck":46,"Golem":47,"Graveler":48,"Grimer":49,"Growlithe":50,"Gyarados":51,"Haunter":52,"Hitmonchan":53,"Hitmonlee":54,"Horsea":55,"Hypno":56,"Ivysaur":57,"Jigglypuff":58,"Jolteon":59,"Jynx":60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,3,94,95,96,97,98,99}
        self.class_map = {"Abra" : 0 ,
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
        self.img_dim = (416, 416) 


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        # img_tensor = torch.from_numpy(img)
        img_tensor = torchvision.transforms.functional.to_tensor(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id



if __name__ == "__main__":
    a = TrainingDataset()
    b,c = a.__getitem__(1)
    b = b.permute(1,2,0)
    print(b.size())
    cv2.imshow("Img", b.numpy())
    cv2.waitKey(10000)