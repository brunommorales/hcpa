import zipfile
from pathlib import Path

# Caminho para o seu arquivo .zip
arquivo_zip = Path('../archive.zip')

# Pasta de destino para extrair os arquivos
destino = Path('data/raw_images/')

# Cria a pasta de destino se ela não existir
destino.mkdir(parents=True, exist_ok=True)

print(f"Descompactando '{arquivo_zip}' para '{destino}'...")

# Abre o arquivo zip e extrai todo o conteúdo
with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
    zip_ref.extractall(destino)

print("Arquivo descompactado com sucesso! ✅")
