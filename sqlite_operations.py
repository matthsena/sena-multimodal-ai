import sqlite3
from sqlite3 import Error
import json


class SQLiteOperations:
    def __init__(self):
        self.conn = sqlite3.connect('data.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_params (
            file_path TEXT PRIMARY KEY,
            features TEXT,
            ocr TEXT,
            panoptic TEXT,
            inception_v3 TEXT,
            resnet50 TEXT
        )
        ''')

    def __del__(self):
        self.conn.close()

    def insert(self, file_path, features, ocr, panoptic, inception_v3, resnet50):
        try:
            features = json.dumps(features)
            ocr = json.dumps(ocr)
            panoptic = json.dumps(panoptic)
            inception_v3 = json.dumps(inception_v3)
            resnet50 = json.dumps(resnet50)
            self.cursor.execute('''
            INSERT INTO image_params (file_path, features, ocr, panoptic, inception_v3, resnet50)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (file_path, features, ocr, panoptic, inception_v3, resnet50))
            self.conn.commit()
        except Error as e:
            print(f"Error in insert: {e}")

    def select(self, file_path):
        try:
            self.cursor.execute('''
        SELECT * FROM image_params WHERE file_path = ?
        ''', (file_path,))
            row = self.cursor.fetchone()
            if row is not None:
                return {
                    'file_path': row[0],
                    'features': json.loads(row[1]),
                    'ocr': json.loads(row[2]),
                    'panoptic': json.loads(row[3]),
                    'inception_v3': json.loads(row[4]),
                    'resnet50': json.loads(row[5])
                }
            else:
                return None
        except Exception as e:
            print(f"Erro ao selecionar dados: {e}")
            return None

    def upsert(self, file_path, features, ocr, panoptic, inception_v3, resnet50):
        try:
            features = json.dumps(features)
            ocr = json.dumps(ocr)
            panoptic = json.dumps(panoptic)
            inception_v3 = json.dumps(inception_v3)
            resnet50 = json.dumps(resnet50)
        except TypeError as e:
            print(f"Erro ao converter os dados para JSON: {e}")
            return

        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO image_params
            (file_path, features, ocr, panoptic, inception_v3, resnet50)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (file_path, features, ocr, panoptic, inception_v3, resnet50))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")

    def upsert_features(self, file_path, features):
        try:
            features = json.dumps(features)
        except TypeError as e:
            print(f"Erro ao converter as características para JSON: {e}")
            return

        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO image_params
            (file_path, features)
            VALUES (?, ?)
            ''', (file_path, features))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")

    def upsert_ocr(self, file_path, ocr):
        try:
            ocr = json.dumps(ocr)
        except TypeError as e:
            print(f"Erro ao converter o OCR para JSON: {e}")
            return

        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO image_params
            (file_path, ocr)
            VALUES (?, ?)
            ''', (file_path, ocr))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")

    def upsert_panoptic(self, file_path, panoptic):
        try:
            panoptic = json.dumps(panoptic)
        except TypeError as e:
            print(f"Erro ao converter o panoptic para JSON: {e}")
            return

        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO image_params
            (file_path, panoptic)
            VALUES (?, ?)
            ''', (file_path, panoptic))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")

    def upsert_inception_v3(self, file_path, inception_v3):
        try:
            inception_v3 = json.dumps(inception_v3)
        except TypeError as e:
            print(f"Erro ao converter o inception_v3 para JSON: {e}")
            return

        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO image_params
            (file_path, inception_v3)
            VALUES (?, ?)
            ''', (file_path, inception_v3))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")

    def upsert_resnet50(self, file_path, resnet50):
        try:
            resnet50 = json.dumps(resnet50)
        except TypeError as e:
            print(f"Erro ao converter o resnet50 para JSON: {e}")
            return

        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO image_params
            (file_path, resnet50)
            VALUES (?, ?)
            ''', (file_path, resnet50))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")
