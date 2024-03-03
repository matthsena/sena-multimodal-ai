import os
import sys


def image_list(lang_path: str):
    try:
        base_path = sys.argv[1]
        folders = [f for f in os.listdir(
            base_path) if os.path.isdir(os.path.join(base_path, f))]
    except IndexError:
        print("Erro: Argumento de linha de comando ausente.")
        return None
    except FileNotFoundError:
        print(f"Erro: O diretório {base_path} não foi encontrado.")
        return None
    except NotADirectoryError:
        print(f"Erro: {base_path} não é um diretório.")
        return None

    for folder in folders:
        try:
            folder_path = os.path.join(base_path, folder)
            folder_path = os.path.join(folder_path, lang_path)
            file_list = [file_name for file_name in os.listdir(
                folder_path) if os.path.isfile(os.path.join(folder_path, file_name))]
        except FileNotFoundError:
            print(f"Erro: O diretório {folder_path} não foi encontrado.")
            continue
        except NotADirectoryError:
            print(f"Erro: {folder_path} não é um diretório.")
            continue

        return (folder_path, sorted(file_list))
