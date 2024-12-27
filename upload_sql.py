import pandas as pd
import pymysql
import json
import csv
import math
import os


def upload_to_sql(file_path, file_name, file_type):

    file = f"{file_path}/{file_name}.{file_type}"  # Excel 檔案名稱與路徑
    if file_type == 'xlsx':
        sheet_name = "Sheet1"  # 指定工作表名稱
        # 讀取 Excel 並轉成 DataFrame
        df = pd.read_excel(file, sheet_name = sheet_name, engine='openpyxl')

        # === 步驟 2: 建立 MySQL 連線 ===
        try:
            connection = pymysql.connect(
            # 連接 MySQL 資料庫
                host = "localhost",   # MySQL 伺服器地址
                user = "root",        # 資料庫使用者帳號
                password = "syntec7750",  # 資料庫密碼
                database = "testdb",    # 資料庫名稱
            )

            with connection.cursor() as cursor:
                print("Successfully connected to MySQL!")

                # === Step 3: Auto-create Table ===
                table_name = f"{file_name}"
                create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ("

                for column, dtype in zip(df.columns, df.dtypes):
                    max_length = df[column].astype(str).apply(len).max()  # 計算欄位最大字串長度
                    if "int" in str(dtype):
                        sql_type = "INT"
                    elif "float" in str(dtype):
                        sql_type = "FLOAT"
                    elif "datetime" in str(dtype):
                        sql_type = "DATETIME"
                    elif max_length > 255:  # 如果欄位長度超過 255，改用 TEXT
                        # print(f'{column}, here')
                        sql_type = "TEXT"
                    else:
                        sql_type = "VARCHAR(255)"
                    create_table_query += f"`{column}` {sql_type}, "
                create_table_query = create_table_query.rstrip(", ") + ");"

                cursor.execute(create_table_query)
                print(f"Table `{table_name}` created successfully!")

                # === Step 4: Insert Data into Table ===
                # for _, row in df.iterrows():
                #     row_values = tuple(None if pd.isna(x) else x for x in row)  # Handle NaN manually
                #     placeholders = ", ".join(["%s"] * len(row))
                #     insert_query = f"INSERT INTO `{table_name}` VALUES ({placeholders})"
                #     cursor.execute(insert_query, row_values)
                # connection.commit()
                # print("Data inserted successfully from Excel to MySQL!")

                # === 讀取舊資料 ===
                cursor.execute(f"SELECT * FROM `{table_name}`")
                old_data = cursor.fetchall()
                old_data_columns = [col[0] for col in cursor.description]

                # 將舊資料轉換為 DataFrame 以便比對
                old_df = pd.DataFrame(old_data, columns=old_data_columns) if old_data else pd.DataFrame()

                # === 插入或更新資料 ===
                for _, row in df.iterrows():
                    row_values = {col: None if pd.isna(val) else val for col, val in row.items()}
                    columns = ", ".join([f"`{col}`" for col in row_values.keys()])
                    placeholders = ", ".join(["%s"] * len(row_values))

                    if not old_df.empty and row_values in old_df.to_dict('records'):
                        update_query = f"UPDATE `{table_name}` SET "
                        update_query += ", ".join([f"`{col}` = %s" for col in row_values.keys()])
                        update_query += f" WHERE `{old_df.columns[0]}` = %s"  # 以第一列作為主鍵
                        cursor.execute(update_query, list(row_values.values()) + [row[old_df.columns[0]]])
                    else:
                        insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
                        cursor.execute(insert_query, tuple(row_values.values()))

                connection.commit()
                print("Data synchronized successfully!")


        except pymysql.MySQLError as e:
            print(f"Error connecting to MySQL: {e}")

        finally:
            if connection:
                connection.close()
                print("MySQL connection closed.")

    elif file_type == 'json':
        # === 步驟 1: 載入 JSON 資料 ===
        try:
            with open(f'{file}', 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("Error: JSON file not found.")
            exit()
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON file.")
            exit()

        # 檢查 JSON 格式，將單一物件包裝成列表
        if isinstance(data, dict):  # 如果資料是單一物件
            data = [data]
        elif not isinstance(data, list):  # 如果資料既不是物件也不是列表
            print("Error: JSON data should be a list of objects or a single object.")
            exit()


        # === 步驟 2: 建立 MySQL 連線 ===
        try:
            connection = pymysql.connect(
                host="localhost",  # MySQL 伺服器地址
                user="root",       # 資料庫使用者帳號
                password="syntec7750",  # 資料庫密碼
                database="testdb",  # 資料庫名稱
            )

            with connection.cursor() as cursor:
                print("Successfully connected to MySQL!")

                # === Step 3: 自動建立資料表 ===
                table_name = f"{file_name}"
                create_table_query = f"CREATE TABLE IF NOT EXISTS `{table_name}` ("

                # 使用第一筆資料的鍵來建立欄位
                first_row = data[0]
                for column, value in first_row.items():
                    if isinstance(value, (list, dict)):
                        sql_type = "LONGTEXT"
                    elif isinstance(value, str) and len(value) > 255:
                        sql_type = "LONGTEXT"
                    elif isinstance(value, str):
                        sql_type = "VARCHAR(255)"
                    else:
                        sql_type = "LONGTEXT"

                    create_table_query += f"`{column}` {sql_type}, "
                create_table_query = create_table_query.rstrip(", ") + ");"

                cursor.execute(create_table_query)
                print(f"Table `{table_name}` created successfully!")

                # === Step 4: 插入資料到資料表 ===
                # for row in data:
                #     row = {
                #         key: None if (isinstance(value, float) and math.isnan(value)) else json.dumps(value) if isinstance(value, (list, dict)) else value
                #         for key, value in row.items()
                #     }
                #     columns = ", ".join([f"`{key}`" for key in row.keys()])
                #     placeholders = ", ".join(["%s"] * len(row))
                #     insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
                #     values = tuple(row.values())

                #     # # 打印插入查询和数据以进行调试
                #     # print("Insert Query:", insert_query)
                #     # print("Values:", values)
                #     cursor.execute(insert_query, values)

                # connection.commit()
                # print("Data inserted successfully from JSON to MySQL!")

                # === 讀取舊資料 ===
                cursor.execute(f"SELECT * FROM `{table_name}`")
                old_data = cursor.fetchall()
                old_data_columns = [col[0] for col in cursor.description]

                old_df = pd.DataFrame(old_data, columns=old_data_columns) if old_data else pd.DataFrame()

                # === 插入或更新資料 ===
                for row in data:
                    row = {
                        key: None if (isinstance(value, float) and math.isnan(value))
                        else json.dumps(value) if isinstance(value, (list, dict))
                        else value
                        for key, value in row.items()
                    }

                    if not old_df.empty and row in old_df.to_dict('records'):
                        # 更新舊資料
                        update_query = f"UPDATE `{table_name}` SET "
                        update_query += ", ".join([f"`{key}` = %s" for key in row.keys()])
                        update_query += f" WHERE `{old_df.columns[0]}` = %s"  # 用第一欄作為主鍵
                        cursor.execute(update_query, list(row.values()) + [row[old_df.columns[0]]])
                    else:
                        # 插入新資料
                        columns = ", ".join([f"`{key}`" for key in row.keys()])
                        placeholders = ", ".join(["%s"] * len(row))
                        insert_query = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
                        cursor.execute(insert_query, tuple(row.values()))

            connection.commit()
            print("Data synchronized successfully from JSON to MySQL!")

        except pymysql.MySQLError as e:
            print(f"Error connecting to MySQL: {e}")

        finally:
            if connection:
                connection.close()
                print("MySQL connection closed.")
    
    else:
        print(file_type)
    return f"上傳完成{file}"


if __name__ == '__main__':
    # === 步驟 1: 讀取 Excel 檔案 ===
    file_path = "auto/data"
    file_list = os.listdir(f'auto/data')
    # print(file_list)

    for tmp_file in file_list:
        # print(tmp_file)
        tmp = tmp_file.split('.')
        file_name = tmp[0]
        file_type = tmp[1]
        upload_to_sql(file_path, file_name, file_type)