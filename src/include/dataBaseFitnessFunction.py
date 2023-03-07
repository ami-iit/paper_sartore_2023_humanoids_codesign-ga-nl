import pandas as pd
import os
import datetime


class DatabaseFitnessFunction:
    def __init__(self, name_database: str) -> None:
        self.name_database_with_ext = name_database + ".csv"
        try:
            self.database = pd.read_csv(self.name_database_with_ext)
        except:
            self.create_empty_csv_database()

    def create_empty_csv_database(self) -> None:
        self.database = self.get_empty_dataframe()
        self.database.to_csv(self.name_database_with_ext, index=False)

    def get_fitness_value(self, chromosome: list) -> float:
        a = self.database["chromosome"]
        if (
            len(self.database["chromosome"]) > 0
            and len(self.database[self.database["chromosome"] == str(chromosome)]) > 0
        ):
            fitness_value = self.database[
                self.database["chromosome"] == str(chromosome)
            ]["fitnessValue"].min()
        else:
            fitness_value = None
        return fitness_value

    def update(
        self,
        chromosome: list,
        fitness_value: float,
    ) -> None:
        timestamp = datetime.datetime.timestamp(datetime.datetime.now())
        df = self.get_empty_dataframe()
        df.loc[len(self.database)] = [timestamp, str(chromosome), fitness_value]
        df.to_csv(self.name_database_with_ext, index=False, mode="a", header=False)

    @staticmethod
    def get_empty_dataframe():
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "chromosome",
                "fitnessValue",
            ]
        )
        return df

    def rename(self, new_name: str) -> None:
        os.rename(self.name_database_with_ext, new_name + ".csv")
        self.name_database_with_ext = new_name + ".csv"
