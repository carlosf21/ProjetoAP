import os
import gzip
import shutil

pasta_atual = os.path.dirname(os.path.abspath(__file__))

def comprimir_ficheiro(nome_ficheiro):
    caminho_original = os.path.join(pasta_atual, nome_ficheiro)
    caminho_comprimido = caminho_original + '.gz'
    
    if not os.path.exists(caminho_original):
        print(f"ERRO: O ficheiro '{nome_ficheiro}' não foi encontrado na pasta:\n{pasta_atual}")
        return

    print(f"A comprimir '{nome_ficheiro}'...")
    
    with open(caminho_original, 'rb') as f_in:
        with gzip.open(caminho_comprimido, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    tam_original = os.path.getsize(caminho_original) / (1024 * 1024)
    tam_comprimido = os.path.getsize(caminho_comprimido) / (1024 * 1024)
    
    print(f"Concluído com sucesso!")
    print(f"   Tamanho Original:   {tam_original:.2f} MB")
    print(f"   Tamanho Comprimido: {tam_comprimido:.2f} MB")
    print(f"   Novo ficheiro gerado: {nome_ficheiro}.gz\n")

comprimir_ficheiro('modelo_numpy_final.pkl')