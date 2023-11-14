import pandas as pd

# CUSTOM DEPENDENCIES
import learning

catalogs = pd.read_excel("Product Catalog.xlsx").astype("str")
catalogs = catalogs.drop_duplicates().reset_index(drop=True)

transactions = pd.read_excel("Product Name from PoS Transactions.xlsx").astype("str")

learning.extract_batch_predict(transactions, catalogs, batch_size=1000)