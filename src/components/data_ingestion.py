import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

import os
import sys
import logging
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataIngestion:
    raw_data_path: str = os.path.join('artifacts', "data.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('dataset/data.csv', encoding="ISO-8859-1")
            logging.info('Read the dataset as dataframe')

            # Create the directory for the raw data file if it doesn't exist
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            # Save the dataframe as CSV to the specified path
            df.to_csv(self.raw_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return self.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    data_path = obj.initiate_data_ingestion()
    print(f"Data ingested and saved to: {data_path}")



