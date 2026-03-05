import os
from PIL import Image
from pillow_heif import register_heif_opener

# Agar Pillow bisa membaca file .heic
register_heif_opener()

SOURCE_DIR = "/teamspace/studios/this_studio/fine-tuning-glm-ocr/raw_data/threads"
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.tiff', '.bmp')

def convert_and_rename():
    # Ambil semua file gambar yang sesuai ekstensi
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(EXTENSIONS)]
    files.sort() # Biar urutannya konsisten
    
    print(f"Ditemukan {len(files)} gambar. Memulai konversi...")

    for i, filename in enumerate(files):
        try:
            file_path = os.path.join(SOURCE_DIR, filename)
            new_name = f"GAMBAR_{i}.jpg"
            target_path = os.path.join(SOURCE_DIR, new_name)

            # Buka gambar
            with Image.open(file_path) as img:
                # Konversi ke RGB (penting untuk PNG/WebP agar tidak error saat simpan ke JPG)
                rgb_img = img.convert('RGB')
                # Simpan sebagai JPEG asli
                rgb_img.save(target_path, "JPEG", quality=95)

            # Hapus file asli jika namanya berbeda dengan yang baru
            if filename != new_name:
                os.remove(file_path)
                
            print(f"Sukses: {filename} -> {new_name}")
        except Exception as e:
            print(f"Gagal memproses {filename}: {e}")

if __name__ == "__main__":
    convert_and_rename()
    print("\nSelesai! Semua gambar kini berformat JPEG asli.")