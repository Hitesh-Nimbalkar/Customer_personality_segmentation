from Customer_personality.exception import ApplicationException
from Customer_personality.logger import logging
from Customer_personality.configuration.configuration import Configuration
from Customer_personality.components.data_ingestion import DataIngestion
import os
from Customer_personality.pipeline.train import Pipeline


def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()

    except Exception as e:
            logging.error(f"{e}")
            print(e)


if __name__ == "__main__":
     main()