import sqlite3
from sqlite3 import Error
import json
import sys

import numpy as np


class SQLiteOperations:
    def __init__(self):
        self.conn = sqlite3.connect('./database/data.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_params (
            file_path TEXT PRIMARY KEY,
            lang TEXT,
            article TEXT,
            features TEXT,
            ocr TEXT,
            panoptic TEXT,
            inception_v3 TEXT,
            resnet50 TEXT
        )
        ''')

    def __del__(self):
        self.conn.close()

    def insert(self, lang, file_path, features, ocr, panoptic, inception_v3, resnet50):
        try:
            features = json.dumps(features)
            ocr = json.dumps(ocr)
            panoptic = json.dumps(panoptic)
            inception_v3 = json.dumps(inception_v3)
            resnet50 = json.dumps(resnet50)
            self.cursor.execute('''
            INSERT INTO image_params (file_path, lang, article, features, ocr, panoptic, inception_v3, resnet50)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, lang, sys.argv[1], features, ocr, panoptic, inception_v3, resnet50))
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
                    'lang': row[1],
                    'article': row[2],
                    'features': json.loads(row[3]),
                    'ocr': json.loads(row[4]),
                    'panoptic': json.loads(row[5]),
                    'inception_v3': json.loads(row[6]),
                    'resnet50': json.loads(row[7])
                }
            else:
                return None
        except Exception as e:
            print(f"Erro ao selecionar dados: {e}")
            return None
        
    def select_by_features(self, features):
        try:
            features = json.dumps(features)
            self.cursor.execute('''
        SELECT * FROM image_params WHERE features = ?
        ''', (features,))
            row = self.cursor.fetchone()
            if row is not None:
                return {
                    'file_path': row[0],
                    'lang': row[1],
                    'article': row[2],
                    'features': json.loads(row[3]),
                    'ocr': json.loads(row[4]),
                    'panoptic': json.loads(row[5]),
                    'inception_v3': json.loads(row[6]),
                    'resnet50': json.loads(row[7])
                }
            else:
                return None
        except Exception as e:
            print(f"Erro ao selecionar dados: {e}")
            return None
    
    def select_feature_by_file_path(self, file_path):
        try:
            self.cursor.execute('''
        SELECT features FROM image_params WHERE file_path = ?
        ''', (file_path,))
            row = self.cursor.fetchone()
            if row is not None:
                # Convert the list back to a numpy array
                return np.array(json.loads(row[0]))
            else:
                return None
        except Exception as e:
            print(f"Erro ao selecionar dados: {e}")
            return None

    def upsert(self, file_path, lang, features, ocr, panoptic, inception_v3, resnet50):
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
            (file_path, lang, article, features, ocr, panoptic, inception_v3, resnet50)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, lang, sys.argv[1], features, ocr, panoptic, inception_v3, resnet50))
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Erro de integridade do SQLite: {e}")
        except sqlite3.ProgrammingError as e:
            print(f"Erro de programação do SQLite: {e}")
        except Exception as e:
            print(f"Erro desconhecido: {e}")

    