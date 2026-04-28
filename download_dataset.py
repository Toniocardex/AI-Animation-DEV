import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm # Questa serve per la barra di caricamento

def download_pixel_art():
    # --- CONFIGURAZIONE ---
    dataset_name = "jbilcke-hf/ai-pixel-art-characters" # Un ottimo dataset di partenza
    output_dir = "data/raw/pretraining_dataset"
    target_size = (64, 64) # La risoluzione che userai per la tua U-Net/VAE
    max_images = 15000     # Inizia con 15k per testare, poi puoi aumentare
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Connessione a Hugging Face per il dataset: {dataset_name}")
    
    try:
        # Scarichiamo il dataset (streaming=True permette di non saturare la RAM)
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        print(f"✅ Download iniziato. Le immagini verranno salvate in: {output_dir}")
        
        count = 0
        for item in tqdm(dataset, total=max_images):
            if count >= max_images:
                break
                
            img = item.get('image') or item.get('img')
            
            if img:
                # 1. Converti in RGBA (gestione trasparenza)
                img = img.convert("RGBA")
                
                # 2. Ridimensiona a 64x64 usando NEAREST per mantenere i pixel netti
                img = img.resize(target_size, Image.NEAREST)
                
                # 3. Salva
                img.save(os.path.join(output_dir, f"pretrain_{count:06d}.png"))
                count += 1
                
        print(f"\n✨ Operazione completata! Scaricate {count} immagini pixel-perfect.")
        
    except Exception as e:
        print(f"❌ Errore durante il download: {e}")
        print("Suggerimento: Assicurati di aver installato le librerie con: pip install datasets tqdm Pillow")

if __name__ == "__main__":
    download_pixel_art()