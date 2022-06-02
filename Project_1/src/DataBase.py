import csv
import os

class DataBase:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def data_entry(self, level_results, fieldnames):
        file_exists = os.path.isfile(self.csv_file)
        try:
            with open(self.csv_file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for data in level_results:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
