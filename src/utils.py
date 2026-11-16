from openpyxl import load_workbook
from openpyxl.styles import Font
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import numpy as np
import os

def format_excel_thresholds(file_path, numeric_df, thresholds_mask, start_col=3):
    """ Excel """
    wb = load_workbook(file_path)
    ws = wb.active
    
    for i, idx in enumerate(numeric_df.index, start=2):
        for j, col in enumerate(numeric_df.columns, start=start_col):
            
            if idx in thresholds_mask.index:
                if thresholds_mask.loc[idx, col]:
                    cell = ws.cell(row=i, column=j)
                    cell.font = Font(bold=True)
    wb.save(file_path)

def calculate_mi_matrix(df, columns):
    """"""
    mi_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    nmi_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    for i in columns:
        for j in columns:
            if i == j:
                valid_data = df[i].dropna()
                if len(valid_data) == 0:
                    mi_matrix.loc[i, j] = 0.0
                    nmi_matrix.loc[i, j] = 1.0
                    continue
                x = y = valid_data.astype(str)
            else:
                valid_data = df[[i, j]].dropna()
                if len(valid_data) == 0:
                    mi_matrix.loc[i, j] = 0.0
                    nmi_matrix.loc[i, j] = 0.0
                    continue
                x = valid_data[i].astype(str)
                y = valid_data[j].astype(str)
            mi = mutual_info_score(x, y)
            nmi = normalized_mutual_info_score(x, y, average_method='arithmetic')
            mi_matrix.loc[i, j] = mi
            nmi_matrix.loc[i, j] = nmi
    return mi_matrix, nmi_matrix
