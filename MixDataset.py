import os
import shutil
import random
from PIL import Image

# Klasör yolları
veri_seti_1_yolu = "C:/TSL Finger Spelling/Images"
veri_seti_2_yolu = "C:/TestForEnsemble"
test_veri_seti_yolu = "C:/MixedDataset"
her_sinif_icin_resim_sayisi = 200
veri_seti_1_orani = 0.75
veri_seti_2_orani = 0.25

# Test veri seti klasörünü oluştur
if not os.path.exists(test_veri_seti_yolu):
    os.makedirs(test_veri_seti_yolu)

def convert_to_png(source_path, dest_path):
    img = Image.open(source_path)
    png_path = os.path.splitext(dest_path)[0] + ".png"
    img.save(png_path)

def process_images(veri_seti_yolu, oran):
    siniflar = {}

    # Dosya isimlerine göre sınıfları ayır
    for dosya in os.listdir(veri_seti_yolu):
        if dosya.endswith('.jpg') or dosya.endswith('.jpeg') or dosya.endswith('.png'):
            sembol = dosya[0].upper()
            if sembol not in siniflar:
                siniflar[sembol] = []
            siniflar[sembol].append(dosya)

    for sembol in siniflar:
        random.shuffle(siniflar[sembol])
        siniflar[sembol] = siniflar[sembol][:int(her_sinif_icin_resim_sayisi * oran)]

    return siniflar

def copy_images_to_test_set(siniflar, veri_seti_yolu):
    for sembol, resimler in siniflar.items():
        for resim in resimler:
            source_path = os.path.join(veri_seti_yolu, resim)
            dest_path = os.path.join(test_veri_seti_yolu, resim)
            if resim.endswith('.jpg') or resim.endswith('.jpeg'):
                convert_to_png(source_path, dest_path)
            else:
                shutil.copy(source_path, dest_path)

# Veri setlerinden resimleri işleyip kopyala
siniflar_veri_seti_1 = process_images(veri_seti_1_yolu, veri_seti_1_orani)
siniflar_veri_seti_2 = process_images(veri_seti_2_yolu, veri_seti_2_orani)

copy_images_to_test_set(siniflar_veri_seti_1, veri_seti_1_yolu)
copy_images_to_test_set(siniflar_veri_seti_2, veri_seti_2_yolu)

print("Test veri seti oluşturuldu.")
